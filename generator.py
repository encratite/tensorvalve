import numpy as np
from keras.utils import Sequence

class SlidingWindowGenerator(Sequence):
	def __init__(self, x_wav, y_wav, input_shape, output_size, batch_size):
		self.x_wav = x_wav
		self.y_wav = y_wav
		self.input_shape = input_shape
		self.input_size = input_shape[0] * input_shape[1]
		self.output_size = output_size
		self.batch_size = batch_size
		assert self.output_size < self.input_size

	def __len__(self):
		samples = len(self.x_wav) - self.input_size + 1
		length = samples // self.batch_size
		return length

	def __getitem__(self, batch_index):
		sample_index = batch_index * self.batch_size
		sample_index_range = range(sample_index, sample_index + self.batch_size)
		get_samples = lambda f: np.array(list(map(f, sample_index_range)))
		x = get_samples(self.get_x)
		y = get_samples(self.get_y)
		return x, y

	def get_x(self, sample_index):
		start = sample_index
		end = start + self.input_size
		x_linear = self.x_wav[start : end]
		x = np.reshape(x_linear, self.input_shape)
		return x

	def get_y(self, sample_index):
		end = sample_index + self.input_size
		start = end  - self.output_size
		y = self.y_wav[start : end]
		return y