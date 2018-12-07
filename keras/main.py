import random
import sys

import numpy as np
import scipy.io.wavfile
from keras import Sequential
from keras.layers import LSTM, GRU, Flatten
from keras.backend import square, mean, sqrt

from profiler import Profiler

def read_wav(path, shape=None):
	print(f'Loading WAV file from "{path}".')
	rate, data = scipy.io.wavfile.read(path)
	return data
	
def reshape_wav(data, sample_shape):
	samples_per_batch = sample_shape[1] * sample_shape[2]
	batches = len(data) // samples_per_batch
	resized_data = np.resize(data, batches * samples_per_batch)
	reshaped_data = np.reshape(resized_data, sample_shape)
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

batch_size = 1
epochs = 10
	
x_training_path = sys.argv[1]
y_training_path = sys.argv[2]

x_validation_path = sys.argv[3]
y_validation_path = sys.argv[4]

profiler = Profiler()

x_training_raw = read_wav(x_training_path)
y_training = read_wav(y_training_path)

x_validation_raw = read_wav(x_validation_path)
y_validation = read_wav(y_validation_path)

profiler.stop('Done loading WAV files.')

if len(x_training_raw) != len(y_training) or len(x_validation_raw) != len(y_validation):
	raise Exception('Dry and wet WAVs must be same length.')

sample_shape = (-1, rnn_timesteps, rnn_features)

x_training = reshape_wav(x_training_raw, sample_shape)
x_validation = reshape_wav(x_validation_raw, sample_shape)
	
model = Sequential()
model.add(LSTM(rnn_units, dropout=rnn_dropout, return_sequences=True, input_shape=(rnn_timesteps, rnn_features)))
model.add(Flatten())
model.compile('adam', loss=root_mean_squared_error)

model.fit(x_training, y_training, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_validation, y_validation))

profiler.stop('Done fitting.')