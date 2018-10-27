import os

from google.protobuf import text_format
import tensorflow as tf

from profiler import Profiler

class TensorValve:
	def __init__(self, layer_type = None, time_steps = None, batch_size = None, input_size = None, layers = None, dropout = None, rnn_bias_initializer = None, activation_function = None, learning_rate = None, time_limit = None, save_path = None):
		assert layer_type != None
		assert batch_size != None
		assert input_size != None
		assert layers != None
		assert save_path != None
		assert dropout != None
		assert activation_function != None
		assert learning_rate != None

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

		self.batch_elements = time_steps * batch_size * input_size

	def train(self, dry_training_wav, wet_training_wav, dry_validation_wav, wet_validation_wav):
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
			loss_minimum = graph.get_tensor_by_name('loss_minimum:0')

			minimize = graph.get_operation_by_name('minimize')
			update_loss_minimum = graph.get_operation_by_name('update_loss_minimum')
			increment_epoch = graph.get_operation_by_name('increment_epoch')

			print('Commencing training.')
			start = time.perf_counter()
			while self.time_limit is None or time.perf_counter() - start < self.time_limit:
				profiler = Profiler()
				operations = [minimize, update_loss_minimum, increment_epoch]
				self.run_operation(dry_training_wav, wet_training_wav, minimize, dry_data_placeholder, wet_data_placeholder, session)
				epoch = session.run(epoch_variable)
				profiler.stop(f'Completed epoch {epoch}.')
				losses = self.run_operation(dry_validation_wav, wet_validation_wav, loss, dry_data_placeholder, wet_data_placeholder, session)
				validation_loss = sum(losses)
				profiler.stop(f'Loss: {validation_loss}')
				if epoch % 10 == 0:
					self.save_session(session, saver)
					profiler.stop('Saved model.')
			self.save_session(session, saver)
			profiler.stop('Training time limit expired. Saved model.')

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
			lstm = self.layer_type(self.layers, self.input_size, dropout = self.dropout, bias_initializer = self.rnn_bias_initializer, name = 'rnn')
			reshaped_dry_data = tf.reshape(dry_data, [self.time_steps, self.batch_size, self.input_size])
			lstm_output, _ = lstm(reshaped_dry_data)
			flat_lstm_output = tf.reshape(lstm_output, batch_shape)
			prediction = self.activation_function(flat_lstm_output, name = 'prediction')
			loss = tf.sqrt(tf.losses.mean_squared_error(prediction, wet_data), name = 'loss')
			loss_minimum = tf.get_variable('loss_minimum', dtype = tf.float)
			update_loss_minimum = tf.minimum(loss, loss_minimum, 'update_loss_minimum')
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