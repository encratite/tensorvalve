import tensorflow as tf

from profiler import Profiler

class LSTMNet:
	def __init__(self, time_steps, batch_count, frame_count, lstm_layers):
		self.time_steps = time_steps
		self.batch_count = batch_count
		self.frame_count = frame_count
		self.lstm_layers = lstm_layers

		self.batch_size = time_steps * batch_count * frame_count

	def train(self, dry_training_wav, wet_training_wav, dry_validation_wav, wet_validation_wav):
		graph = self.get_graph()
		with tf.Session(graph = graph) as session:
			initializer = tf.global_variables_initializer()
			session.run(initializer)
			iteration = 1
			dry_data_placeholder = graph.get_tensor_by_name('dry_data:0')
			wet_data_placeholder = graph.get_tensor_by_name('wet_data:0')
			loss = graph.get_tensor_by_name('loss:0')
			minimize = graph.get_operation_by_name('minimize')
			print('Commencing training.')
			while True:
				print(f'Iteration {iteration}')
				self.run_operation(dry_training_wav, wet_training_wav, minimize, dry_data_placeholder, wet_data_placeholder, session)
				losses = self.run_operation(dry_validation_wav, wet_validation_wav, loss, dry_data_placeholder, wet_data_placeholder, session)
				validation_loss = sum(losses)
				print(f'Validation: {validation_loss}')
				iteration += 1

	def get_graph(self):
		graph = tf.Graph()
		with graph.as_default():
			batch_shape = [self.batch_size]
			dry_data = tf.placeholder(tf.float32, batch_shape, 'dry_data')
			wet_data = tf.placeholder(tf.float32, batch_shape, 'wet_data')

			lstm = tf.contrib.cudnn_rnn.CudnnLSTM(self.lstm_layers, self.frame_count, name = 'lstm')
			reshaped_dry_data = tf.reshape(dry_data, [self.time_steps, self.batch_count, self.frame_count])
			lstm_output, _ = lstm(reshaped_dry_data)
			flat_lstm_output = tf.reshape(lstm_output, batch_shape)
			prediction = tf.nn.elu(flat_lstm_output)

			loss = tf.sqrt(tf.losses.mean_squared_error(prediction, wet_data), name = 'loss')
			optimizer = tf.train.AdamOptimizer()
			minimize = optimizer.minimize(loss, name = 'minimize')
		return graph

	def run_operation(self, dry_data, wet_data, operation, dry_data_placeholder, wet_data_placeholder, session):
		offset = 0
		output = []
		profiler = Profiler()
		while offset + self.batch_size < len(dry_data):
			print(f'Progress: {offset}/{len(dry_data)}')
			dry_batch = self.get_batch(dry_data, offset)
			wet_batch = self.get_batch(wet_data, offset)
			feed = {
				dry_data_placeholder: dry_batch,
				wet_data_placeholder: wet_batch
			}
			profiler.stop('overhead')
			operation_output = session.run(operation, feed)
			profiler.stop('kernel')
			output.append(operation_output)
			offset += self.batch_size
		return output

	def get_batch(self, data, offset):
		return data[offset : offset + self.batch_size]