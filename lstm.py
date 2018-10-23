import os

from google.protobuf import text_format
import tensorflow as tf

from profiler import Profiler

class LSTMNet:
	def __init__(self, time_steps, batch_count, frame_count, lstm_layers, graph_file):
		self.time_steps = time_steps
		self.batch_count = batch_count
		self.frame_count = frame_count
		self.lstm_layers = lstm_layers

		self.graph_file = graph_file

		self.batch_size = time_steps * batch_count * frame_count

	def train(self, dry_training_wav, wet_training_wav, dry_validation_wav, wet_validation_wav):
		graph = self.get_graph()
		profiler = Profiler()
		with tf.Session(graph = graph) as session:
			#initializer = tf.global_variables_initializer()
			#session.run(initializer)
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
				self.save_graph(graph)

	def get_graph(self):
		graph = tf.Graph()
		with graph.as_default():
			if os.path.isfile(self.graph_file):
				self.load_graph()
			else:
				print('Constructing graph.')
				profiler = Profiler()
				epoch_variable = tf.get_variable('epoch', dtype = tf.int32, initializer = tf.constant(0))
				increment_epoch = tf.assign(epoch_variable, epoch_variable + 1, name = 'increment_epoch')
				batch_shape = [self.batch_size]
				dry_data = tf.placeholder(tf.float32, batch_shape, 'dry_data')
				wet_data = tf.placeholder(tf.float32, batch_shape, 'wet_data')
				lstm = tf.contrib.cudnn_rnn.CudnnLSTM(self.lstm_layers, self.frame_count, bias_initializer = tf.contrib.layers.xavier_initializer(), name = 'lstm')
				reshaped_dry_data = tf.reshape(dry_data, [self.time_steps, self.batch_count, self.frame_count])
				lstm_output, _ = lstm(reshaped_dry_data)
				flat_lstm_output = tf.reshape(lstm_output, batch_shape)
				prediction = tf.nn.elu(flat_lstm_output)
				loss = tf.sqrt(tf.losses.mean_squared_error(prediction, wet_data), name = 'loss')
				optimizer = tf.train.AdamOptimizer()
				minimize = optimizer.minimize(loss, name = 'minimize')
				profiler.stop('Done constructing graph.')
				self.save_graph(graph)
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

	def load_graph(self):
		profiler = Profiler()
		with open(self.graph_file, 'rb') as file:
			content = file.read()
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(content)
		tf.import_graph_def(graph_def)
		profiler.stop(f'Restored graph from "{self.graph_file}".')

	def save_graph(self, graph):
		profiler = Profiler()
		directory = os.path.dirname(self.graph_file)
		file = os.path.basename(self.graph_file)
		tf.train.write_graph(graph, directory, file, False)
		profiler.stop(f'Wrote graph to "{self.graph_file}".')