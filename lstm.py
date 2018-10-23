import os

from google.protobuf import text_format
import tensorflow as tf

from profiler import Profiler

class LSTMNet:
	def __init__(self, time_steps, batch_count, frame_count, lstm_layers, graph_file, checkpoint_file):
		self.time_steps = time_steps
		self.batch_count = batch_count
		self.frame_count = frame_count
		self.lstm_layers = lstm_layers

		self.graph_file = graph_file
		self.checkpoint_file = checkpoint_file

		self.batch_size = time_steps * batch_count * frame_count

	def train(self, dry_training_wav, wet_training_wav, dry_validation_wav, wet_validation_wav):
		graph = self.get_graph()
		profiler = Profiler()
		with tf.Session(graph = graph) as session:
			saver = tf.train.Saver()
			if os.path.isfile(self.checkpoint_file):
				print(f'Restoring session from checkpoint "{self.checkpoint_file}".')
				saver.restore(session, self.checkpoint_file)
				profiler.stop(f'Done restoring session.')
			initializer = tf.global_variables_initializer()
			session.run(initializer)
			dry_data_placeholder = graph.get_tensor_by_name('dry_data:0')
			wet_data_placeholder = graph.get_tensor_by_name('wet_data:0')
			loss = graph.get_tensor_by_name('loss:0')
			minimize = graph.get_operation_by_name('minimize')
			print('Commencing training.')
			iteration = 1
			while True:
				print(f'Iteration {iteration}')
				profiler = Profiler()
				self.run_operation(dry_training_wav, wet_training_wav, minimize, dry_data_placeholder, wet_data_placeholder, session)
				losses = self.run_operation(dry_validation_wav, wet_validation_wav, loss, dry_data_placeholder, wet_data_placeholder, session)
				validation_loss = sum(losses)
				profiler.stop(f'Loss: {validation_loss}')
				saver.save(session, self.checkpoint_file)
				profiler.stop('Saved checkpoint.')
				iteration += 1

	def get_graph(self):
		graph = tf.Graph()
		with graph.as_default():
			profiler = Profiler()
			if os.path.isfile(self.graph_file):
				print(f'Restoring graph from "{self.graph_file}".')
				with open(self.graph_file, 'rb') as file:
					content = file.read()
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(content)
				tf.import_graph_def(graph_def)
				profiler.stop('Done loading graph.')
			else:
				print('Constructing graph.')
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
				directory = os.path.dirname(self.graph_file)
				file = os.path.basename(self.graph_file)
				tf.train.write_graph(graph, directory, file, False)
				profiler.stop(f'Constructed graph and stored it in "{self.graph_file}".')
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