import os

from google.protobuf import text_format
import tensorflow as tf

from profiler import Profiler

class LSTMNet:
	def __init__(self, time_steps, batch_count, frame_count, lstm_layers, save_path):
		self.time_steps = time_steps
		self.batch_count = batch_count
		self.frame_count = frame_count
		self.lstm_layers = lstm_layers

		prefix = os.path.basename(save_path)
		self.save_path = os.path.join(save_path, prefix)
		self.save_path_directory = save_path

		self.batch_size = time_steps * batch_count * frame_count

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
			increment_epoch = graph.get_operation_by_name('increment_epoch')
			dry_data_placeholder = graph.get_tensor_by_name('dry_data:0')
			wet_data_placeholder = graph.get_tensor_by_name('wet_data:0')
			loss = graph.get_tensor_by_name('loss:0')
			minimize = graph.get_operation_by_name('minimize')
			print('Commencing training.')
			while True:
				profiler = Profiler()
				self.run_operation(dry_training_wav, wet_training_wav, minimize, dry_data_placeholder, wet_data_placeholder, session)
				session.run(increment_epoch)
				epoch = session.run(epoch_variable)
				profiler.stop(f'Completed epoch {epoch}.')
				losses = self.run_operation(dry_validation_wav, wet_validation_wav, loss, dry_data_placeholder, wet_data_placeholder, session)
				validation_loss = sum(losses)
				profiler.stop(f'Loss: {validation_loss}')
				self.save_session(session, saver)
				profiler.stop('Saved model.')

	def get_graph(self):
		graph = tf.Graph()
		if self.save_path_exists():
			return graph
		with graph.as_default():
			print('Constructing graph.')
			profiler = Profiler()
			batch_shape = [self.batch_size]
			dry_data = tf.placeholder(tf.float32, batch_shape, 'dry_data')
			wet_data = tf.placeholder(tf.float32, batch_shape, 'wet_data')
			epoch_variable = tf.get_variable('epoch', dtype = tf.int32, initializer = tf.constant(0))
			increment_epoch = tf.assign(epoch_variable, epoch_variable + 1, name = 'increment_epoch')
			lstm = tf.contrib.cudnn_rnn.CudnnLSTM(self.lstm_layers, self.frame_count, bias_initializer = tf.contrib.layers.xavier_initializer(), name = 'lstm')
			reshaped_dry_data = tf.reshape(dry_data, [self.time_steps, self.batch_count, self.frame_count])
			lstm_output, _ = lstm(reshaped_dry_data)
			flat_lstm_output = tf.reshape(lstm_output, batch_shape)
			prediction = tf.nn.elu(flat_lstm_output)
			loss = tf.sqrt(tf.losses.mean_squared_error(prediction, wet_data), name = 'loss')
			optimizer = tf.train.AdamOptimizer()
			minimize = optimizer.minimize(loss, name = 'minimize')
			profiler.stop('Done constructing graph.')
		return graph

	def run_operation(self, dry_data, wet_data, operation, dry_data_placeholder, wet_data_placeholder, session):
		offset = 0
		output = []
		while offset + self.batch_size < len(dry_data):
			dry_batch = self.get_batch(dry_data, offset)
			wet_batch = self.get_batch(wet_data, offset)
			feed = {
				dry_data_placeholder: dry_batch,
				wet_data_placeholder: wet_batch
			}
			operation_output = session.run(operation, feed)
			output.append(operation_output)
			offset += self.batch_size
		return output

	def get_batch(self, data, offset):
		return data[offset : offset + self.batch_size]

	def save_path_exists(self):
		return os.path.exists(self.save_path_directory)

	def save_session(self, session, saver):
		if not self.save_path_exists():
			os.makedirs(self.save_path_directory)
		saver.save(session, self.save_path)