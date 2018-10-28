import os

from google.protobuf import text_format
import tensorflow as tf

from profiler import Profiler
from modelinfo import ModelInfo

class TensorValve:
	def __init__(
		self,
		name = None,
		layer_type = None,
		time_steps = None,
		batch_size = None,
		input_size = None,
		layers = None,
		dropout = None,
		rnn_bias_initializer = None,
		activation_function = None,
		learning_rate = None,
		time_limit = None,
		save_path = None,
		database = None
	):
		assert name is not None
		assert layer_type is not None
		assert batch_size is not None
		assert input_size is not None
		assert layers is not None
		assert save_path is not None
		assert dropout is not None
		assert activation_function is not None
		assert learning_rate is not None
		assert database is not None

		self.name = name

		self.layer_type = layer_type
		self.time_steps = time_steps
		self.batch_size = batch_size
		self.input_size = input_size
		self.layers = layers
		self.dropout = dropout
		self.rnn_bias_initializer = rnn_bias_initializer
		self.activation_function = activation_function

		self.learning_rate = learning_rate

		self.time_limit = time_limit
		prefix = os.path.basename(save_path)
		self.save_path = os.path.join(save_path, prefix)
		self.save_path_directory = save_path
		self.database = database

		self.batch_elements = time_steps * batch_size * input_size

	def train(self, dry_training_wav, wet_training_wav, dry_validation_wav, wet_validation_wav):
		epoch = None
		minimum_loss = None
		try:
			graph = self.get_graph()
			profiler = Profiler()
			with tf.Session(graph = graph) as session:
				initializer = tf.global_variables_initializer()
				session.run(initializer)
				saver = tf.train.Saver()
				if self.save_path_exists():
					saver.restore(session, self.save_path)
					profiler.stop('Restored model.')

				epoch_variable = graph.get_tensor_by_name('epoch:0')
				dry_data_placeholder = graph.get_tensor_by_name('dry_data:0')
				wet_data_placeholder = graph.get_tensor_by_name('wet_data:0')
				loss = graph.get_tensor_by_name('loss:0')
				minimum_loss = graph.get_tensor_by_name('minimum_loss:0')
				maximum = graph.get_tensor_by_name('maximum:0')

				minimize = graph.get_operation_by_name('minimize')
				update_minimum_loss = graph.get_operation_by_name('update_minimum_loss')
				increment_epoch = graph.get_operation_by_name('increment_epoch')

				print('Commencing training.')
				start = time.perf_counter()
				while self.time_limit is None or time.perf_counter() - start < self.time_limit:
					profiler = Profiler()
					operations = [minimize, update_minimum_loss, increment_epoch]
					self.run_operation(dry_training_wav, wet_training_wav, minimize, dry_data_placeholder, wet_data_placeholder, session)
					epoch = session.run(epoch_variable)
					minimum_loss = session.run(minimum_loss)
					profiler.stop(f'Completed epoch {epoch}.')
					losses = self.run_operation(dry_validation_wav, wet_validation_wav, loss, dry_data_placeholder, wet_data_placeholder, session)
					validation_loss = sum(losses)
					profiler.stop(f'Loss: {validation_loss}')
					print(f'Maximum: {session.run(maximum)}')
					if epoch % 10 == 0:
						self.save_session(session, saver)
						profiler.stop('Saved model.')
					self.upate_database(epoch, minimum_loss, False)
				self.save_session(session, saver)
				self.upate_database(epoch, minimum_loss, True)
				profiler.stop('Training time limit expired. Saved model.')
		except Exception as error:
			print(f'An error occurred while training "{self.name}": {error}')
			self.update_database(epoch, minimum_loss, False, str(error))

	def get_graph(self):
		graph = tf.Graph()
		with graph.as_default():
			print('Constructing graph.')
			profiler = Profiler()
			batch_shape = [self.batch_elements]
			dry_data = tf.placeholder(tf.float32, batch_shape, 'dry_data')
			wet_data = tf.placeholder(tf.float32, batch_shape, 'wet_data')
			epoch_variable = tf.get_variable('epoch', dtype = tf.int32, initializer = tf.constant(0))
			increment_epoch = tf.assign(epoch_variable, epoch_variable + 1, name = 'increment_epoch')
			rnn = self.layer_type(self.layers, self.input_size, dropout = self.dropout, bias_initializer = self.rnn_bias_initializer, name = 'rnn')
			reshaped_dry_data = tf.reshape(dry_data, [self.time_steps, self.batch_size, self.input_size])
			rnn_output, _ = rnn(reshaped_dry_data)
			maximum = tf.reduce_max(rnn_output, name = 'maximum')
			flat_rnn_output = tf.reshape(rnn_output, batch_shape)
			prediction = self.activation_function(flat_rnn_output, name = 'prediction')
			loss = tf.sqrt(tf.losses.mean_squared_error(prediction, wet_data), name = 'loss')
			minimum_loss = tf.get_variable('minimum_loss', dtype = tf.float32)
			update_minimum_loss = tf.minimum(loss, minimum_loss, 'update_minimum_loss')
			optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
			minimize = optimizer.minimize(loss, name = 'minimize')
			profiler.stop('Done constructing graph.')
		return graph

	def run_operation(self, dry_data, wet_data, operation, dry_data_placeholder, wet_data_placeholder, session):
		offset = 0
		output = []
		while offset + self.batch_elements < len(dry_data):
			dry_batch = self.get_batch(dry_data, offset)
			wet_batch = self.get_batch(wet_data, offset)
			feed = {
				dry_data_placeholder: dry_batch,
				wet_data_placeholder: wet_batch
			}
			operation_output = session.run(operation, feed)
			output.append(operation_output)
			offset += self.batch_elements
		return output

	def get_batch(self, data, offset):
		return data[offset : offset + self.batch_elements]

	def save_path_exists(self):
		return os.path.exists(self.save_path_directory)

	def save_session(self, session, saver):
		if not self.save_path_exists():
			os.makedirs(self.save_path_directory)
		saver.save(session, self.save_path)

	def update_database(self, epochs, minimum_loss, done_training = None, error = None):
		model_info = self.database.get_model_info(self.name)
		if model_info is None:
			model_info = ModelInfo(
				self.name,
				epochs,
				minimum_loss,
				done_training,
				error
			)
		else:
			model_info.epochs = epochs
			model_info.minimum_loss = minimum_loss
			model_info.done_training = done_training
			model_info.error = error
		self.database.set_model_info(model_info)