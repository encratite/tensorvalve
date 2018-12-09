import numpy as np
import scipy.io.wavfile

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
