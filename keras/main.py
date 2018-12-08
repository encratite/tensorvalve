import os
import random
import sys
import time

if len(sys.argv) != 7:
	print('Usage:')
	print(f'{sys.argv[0]} <dry training WAV file> <wet training WAV file> <dry validation WAV file> <wet validation WAV file> <database path> <GPU|CPU>')
	sys.exit(1)

x_training_path = sys.argv[1]
y_training_path = sys.argv[2]

x_validation_path = sys.argv[3]
y_validation_path = sys.argv[4]

database_path = sys.argv[5]

mode = sys.argv[6]

if mode == 'CPU':
	os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import numpy as np
import scipy.io.wavfile
from keras import Sequential
from keras.layers import LSTM, GRU, Flatten, Reshape
from keras.backend import square, mean, sqrt

from profiler import Profiler
from timelimit import TimeLimitCallback
from database import ModelDatabase

def read_wav(path):
	print(f'Loading WAV file from "{path}".')
	rate, data = scipy.io.wavfile.read(path)
	return data
	
def reshape_wav(data, input_shape):
	samples_per_batch = input_shape[0] * input_shape[1]
	batches = len(data) // samples_per_batch
	resized_data = np.resize(data, batches * samples_per_batch)
	reshaped_data = np.reshape(resized_data, (-1, ) + input_shape)
	return reshaped_data
	
def root_mean_squared_error(y_true, y_pred):
        return sqrt(mean(square(y_pred - y_true), axis=-1))

def train(x_training_wav, y_training_wav, x_validation_wav, y_validation_wav, options):
	rnn_types = {
		'LSTM': LSTM,
		'GRU': GRU
	}
	rnn_type_key = options['rnn_type']
	rnn_type = rnn_types[rnn_type_key]
	
	rnn_dropout = options['rnn_dropout']
	rnn_units = options['rnn_units']
	rnn_timesteps = options['rnn_timesteps']
	rnn_features = rnn_units

	batch_size = options['batch_size']
	epochs = 1000000
	time_limit = options['time_limit']

	profiler = Profiler()

	input_shape = (rnn_timesteps, rnn_features)
	
	x_training = reshape_wav(x_training_wav, input_shape)
	y_training = reshape_wav(y_training_wav, input_shape)
	
	x_validation = reshape_wav(x_validation_wav, input_shape)
	y_validation = reshape_wav(y_validation_wav, input_shape)
		
	model = Sequential()
	model.add(rnn_type(rnn_units, dropout=rnn_dropout, return_sequences=True, input_shape=input_shape))
	model.compile('adam', loss=root_mean_squared_error)

	time_limit_callback = TimeLimitCallback(time_limit)
	history = model.fit(x_training, y_training, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_validation, y_validation), callbacks=[time_limit_callback])

	profiler.stop('Done training.')

	loss = history.history['loss'][-1]
	val_loss = history.history['val_loss'][-1]

	return (loss, val_loss)

profiler = Profiler()

x_training_wav = read_wav(x_training_path)
y_training_wav = read_wav(y_training_path)

x_validation_wav = read_wav(x_validation_path)
y_validation_wav = read_wav(y_validation_path)

profiler.stop('Done loading WAV files.')

if len(x_training_wav) != len(y_training_wav) or len(x_validation_wav) != len(y_validation_wav):
	raise Exception('Dry and wet WAVs must be same length.')

option_values = {
	'mode': [mode],
	'rnn_type': ['LSTM', 'GRU'],
	'rnn_dropout': [0, 0.25, 0.5],
	'rnn_units': [32, 64, 128, 256],
	'rnn_timesteps': [16, 32, 64, 128, 256],
	'batch_size': [16, 32, 64, 128, 256],
	'time_limit': [15 * 60]
}

with ModelDatabase(database_path) as database:
	while True:
		try:
			print('Options:')
			options = {}
			for key in option_values:
				values = option_values[key]
				value = random.choice(values)
				options[key] = value
				print(f'{key} = {value}')
			if database.model_info_exists(options):
				print('Skipping permutation.')
				time.sleep(1)
				continue
			loss, val_loss = train(x_training_wav, y_training_wav, x_validation_wav, y_validation_wav, options)
			values = dict(options)
			values['loss'] = loss
			values['val_loss'] = val_loss
			database.save_model_info(values)
		except Exception as error:
			print(error)
			time.sleep(5)
			continue