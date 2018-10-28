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
	layer_types = ('lt', [('lstm', tf.contrib.cudnn_rnn.CudnnLSTM), ('gru', tf.contrib.cudnn_rnn.CudnnGRU)])
	time_steps = ('ts', [1, 4, 16, 64])
	batch_sizes = ('bs', [1, 4, 16, 64, 256, 512])
	frame_counts = ('fc', [32, 64, 128, 256])
	layers = ('la', [1, 8, 64, 128])
	dropouts = ('dr', [0.0, 0.25, 0.5])
	rnn_bias_initializers = ('bi', [('zero', None), ('xavier', tf.contrib.layers.xavier_initializer())])
	activation_functions = ('af', [('elu', tf.nn.elu), ('softsign', tf.nn.softsign), ('tanh', tf.tanh)])
	learning_rates = ('lr', [0.001, 0.005, 0.01, 0.05])
	parameter_definitions = [
		layer_types,
		time_steps,
		batch_sizes,
		frame_counts,
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
		model_info = database.get_model(name)
		if model_info is not None and model_info.done_training:
			print(f'Skipping "{name}" because it has already been trained.')
			continue
		print(f'Creating trainer for "{name}".')
		trainer = TensorValve(
			name = name,
			layer_type = permutation['lt'],
			time_steps = permutation['ts'],
			batch_size = permutation['bs'],
			input_size = permutation['is'],
			layers = permutation['la'],
			dropout = permutation['dr'],
			rnn_bias_initializer = permutation['bi'],
			activation_function = permutation['af'],
			learning_rate = permutation['lr'],
			time_limit = time_limit,
			save_path = name,
			database = database
		)
		trainer.train(dry_training_wav, wet_training_wav, dry_validation_wav, wet_validation_wav)