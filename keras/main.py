import os
import random
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import scipy.io.wavfile
from keras import Sequential
from keras.layers import LSTM, GRU, Flatten, Reshape
from keras.backend import square, mean, sqrt

from profiler import Profiler

def read_wav(path, input_shape):
	print(f'Loading WAV file from "{path}".')
	rate, data = scipy.io.wavfile.read(path)
	reshaped_data = reshape_wav(data, input_shape)
	return reshaped_data
	
def reshape_wav(data, input_shape):
	samples_per_batch = input_shape[0] * input_shape[1]
	batches = len(data) // samples_per_batch
	resized_data = np.resize(data, batches * samples_per_batch)
	reshaped_data = np.reshape(resized_data, (-1, ) + input_shape)
	return reshaped_data
	
def root_mean_squared_error(y_true, y_pred):
        return sqrt(mean(square(y_pred - y_true), axis=-1))
	
if len(sys.argv) != 5:
	print('Usage:')
	print(f'{sys.argv[0]} <dry training WAV file> <wet training WAV file> <dry validation WAV file> <wet validation WAV file>')
	sys.exit(1)

rnn_dropout = 0.25
rnn_units = 96
rnn_timesteps = 64
rnn_features = rnn_units

batch_size = 128
epochs = 10
	
x_training_path = sys.argv[1]
y_training_path = sys.argv[2]

x_validation_path = sys.argv[3]
y_validation_path = sys.argv[4]

profiler = Profiler()

input_shape = (rnn_timesteps, rnn_features)

x_training = read_wav(x_training_path, input_shape)
y_training = read_wav(y_training_path, input_shape)

x_validation = read_wav(x_validation_path, input_shape)
y_validation = read_wav(y_validation_path, input_shape)

profiler.stop('Done loading WAV files.')

if len(x_training) != len(y_training) or len(x_validation) != len(y_validation):
	raise Exception('Dry and wet WAVs must be same length.')
	
model = Sequential()
model.add(LSTM(rnn_units, dropout=rnn_dropout, return_sequences=True, input_shape=input_shape))
model.compile('adam', loss=root_mean_squared_error)

model.fit(x_training, y_training, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_validation, y_validation))

profiler.stop('Done training.')