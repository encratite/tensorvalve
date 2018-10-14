import sys

import scipy.io.wavfile
import tensorflow as tf

def read_wav(path):
	rate, data = scipy.io.wavfile.read(path)
	return tf.convert_to_tensor(data, dtype = tf.float32)

def get_batch(data, offset, batch_size):
	return data[offset : offset + batch_size]

def get_length(tensor):
	return tensor.shape[0].value

def run_operation(dry_data, wet_data, batch_size, operation, session):
	offset = 0
	output = []
	while offset + batch_size < get_length(dry_training_wav):
		dry_batch = get_batch(dry_data, offset, batch_size)
		wet_batch = get_batch(wet_data, offset, batch_size)
		feed = {
			dry_data: dry_batch,
			wet_data: wet_batch
		}
		operation_output = session.run(operation, feed)
		output.append(operation_output)
		offset += batch_size
	return output

def get_graph():
	graph = tf.Graph()
	with graph.as_default():
		frame_count = 96
		lstm_layers = 64

		frame_shape = [frame_count]

		dry_data = tf.placeholder(tf.float32, frame_shape, 'dry_data')
		wet_data = tf.placeholder(tf.float32, frame_shape, 'wet_data')

		lstm = tf.contrib.cudnn_rnn.CudnnLSTM(lstm_layers, frame_count)
		reshaped_dry_data = tf.reshape(dry_data, [1, 1, frame_count])
		lstm_output, _ = lstm(reshaped_dry_data)
		flat_lstm_output = tf.reshape(lstm_output, frame_shape)
		prediction = tf.nn.elu(flat_lstm_output)

		loss = tf.sqrt(tf.losses.mean_squared_error(prediction, wet_data), name = 'loss')
		optimizer = tf.train.AdamOptimizer()
		minimize = optimizer.minimize(loss, name = 'minimize')
	return graph

def train(dry_training_wav, wet_training_wav, dry_validation_wav, wet_validation_wav):
	graph = get_graph()
	with tf.Session(graph = graph) as session:
		initializer = tf.global_variables_initializer()
		session.run(initializer)
		iteration = 1
		dry_data_placeholder = graph.get_operation_by_name('dry_data')
		batch_size = dry_data_placeholder.outputs[0].shape[0].value
		loss = graph.get_operation_by_name('loss')
		minimize = graph.get_operation_by_name('minimize')
		print('Commencing training.')
		while True:
			print(f'Iteration {iteration}')
			run_operation(dry_training_wav, wet_training_wav, batch_size, minimize, session)
			losses = run_operation(dry_validation_wav, wet_validation_wav, batch_size, loss, session)
			validation_loss = sum(losses)
			print(f'Validation: {validation_loss}')
			iteration += 1

if len(sys.argv) != 5:
	print('Usage:')
	print(f'{sys.argv[0]} <dry training WAV file> <wet training WAV file> <dry validation WAV file> <wet validation WAV file>')
	sys.exit(1)

dry_training_wav_path = sys.argv[1]
wet_training_wav_path = sys.argv[2]

dry_validation_wav_path = sys.argv[3]
wet_validation_wav_path = sys.argv[4]

dry_training_wav = read_wav(dry_training_wav_path)
wet_training_wav = read_wav(wet_training_wav_path)

dry_validation_wav = read_wav(dry_validation_wav_path)
wet_validation_wav = read_wav(wet_validation_wav_path)

if get_length(dry_training_wav) != get_length(wet_training_wav) or get_length(dry_validation_wav) != get_length(wet_validation_wav):
	raise Exception('Dry and wet WAVs must be same length.')

train(dry_training_wav, wet_training_wav, dry_validation_wav, wet_validation_wav)