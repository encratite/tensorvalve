import random
import sys

import scipy.io.wavfile
import tensorflow as tf

from tensorvalve import TensorValve
from database import TensorValveDatabase
from profiler import Profiler

def read_wav(path):
	print(f'Loading WAV file from "{path}".')
	rate, data = scipy.io.wavfile.read(path)
	return data

def get_parameter_permutation():
	# layer_types = ('layer_type', [('lstm', tf.contrib.cudnn_rnn.CudnnLSTM), ('gru', tf.contrib.cudnn_rnn.CudnnGRU)])
	layer_types = ('layer_type', [('lstm', tf.contrib.cudnn_rnn.CudnnLSTM)])
	# time_steps = ('time_steps', [1, 4, 16, 64])
	time_steps = ('time_steps', [32])
	# batch_sizes = ('batch_size', [1, 4, 16, 64, 256, 512])
	batch_sizes = ('batch_size', [128])
	# input_sizes = ('input_size', [32, 64, 128, 256])
	input_sizes = ('input_size', [96])
	# layers = ('layers', [1, 8, 64, 128])
	layers = ('layers', [64])
	dropouts = ('dropouts', [0.0, 0.25, 0.5])
	rnn_bias_initializers = ('bias_initializer', [('zero', None), ('xavier', tf.contrib.layers.xavier_initializer())])
	activation_functions = ('activation_function', [('elu', tf.nn.elu), ('softsign', tf.nn.softsign), ('tanh', tf.tanh)])
	learning_rates = ('learning_rate', [0.001, 0.005, 0.01, 0.05])
	parameter_definitions = [
		layer_types,
		time_steps,
		batch_sizes,
		input_sizes,
		layers,
		dropouts,
		rnn_bias_initializers,
		activation_functions,
		learning_rates
	]
	name_tokens = []
	permutation = {}
	for key, values in parameter_definitions:
		parameter_value = random.choice(values)
		if type(parameter_value) is tuple:
			description = parameter_value[0]
			value = parameter_value[1]
		else:
			description = str(parameter_value)
			value = parameter_value
		name_tokens.append(description)
		permutation[key] = value
	name = '-'.join(name_tokens)
	return (name, permutation)

if len(sys.argv) != 6:
	print('Usage:')
	print(f'{sys.argv[0]} <dry training WAV file> <wet training WAV file> <dry validation WAV file> <wet validation WAV file> <database file>')
	sys.exit(1)

dry_training_wav_path = sys.argv[1]
wet_training_wav_path = sys.argv[2]

dry_validation_wav_path = sys.argv[3]
wet_validation_wav_path = sys.argv[4]

database_path = sys.argv[5]

profiler = Profiler()

dry_training_wav = read_wav(dry_training_wav_path)
wet_training_wav = read_wav(wet_training_wav_path)

dry_validation_wav = read_wav(dry_validation_wav_path)
wet_validation_wav = read_wav(wet_validation_wav_path)

profiler.stop('Done loading WAV files.')

if len(dry_training_wav) != len(wet_training_wav) or len(dry_validation_wav) != len(wet_validation_wav):
	raise Exception('Dry and wet WAVs must be same length.')

time_limit = 15 * 60

with TensorValveDatabase(database_path) as database:
	while True:
		name, permutation = get_parameter_permutation()
		model_info = database.get_model_info(name)
		if model_info is not None and model_info.done_training:
			print(f'Skipping "{name}" because it has already been trained.')
			continue
		print(f'Creating trainer for "{name}".')
		trainer = TensorValve(
			name = name,
			layer_type = permutation['layer_type'],
			time_steps = permutation['time_steps'],
			batch_size = permutation['batch_size'],
			input_size = permutation['input_size'],
			layers = permutation['layers'],
			dropout = permutation['dropouts'],
			rnn_bias_initializer = permutation['bias_initializer'],
			activation_function = permutation['activation_function'],
			learning_rate = permutation['learning_rate'],
			time_limit = time_limit,
			save_path = name,
			database = database
		)
		trainer.train(dry_training_wav, wet_training_wav, dry_validation_wav, wet_validation_wav)