import numpy as np
from keras.utils import Sequence

class SlidingWindowGenerator(Sequence):
	def __init__(self, x_wav, y_wav, input_shape, output_size):
		self.x_wav = x_wav
		self.y_wav = y_wav
		self.input_shape = input_shape
		self.input_size = input_shape[0] * input_shape[1]
		self.output_size = output_size
		assert self.output_size < self.input_size

	def __len__(self):
		length = len(self.x_wav) - self.input_size + 1
		return length

	def __getitem__(self, index):
		y_offset = index + self.input_size
		x_linear = self.x_wav[index : y_offset]
		x = np.reshape(x_linear, self.input_shape)
		y = self.y_wav[y_offset  - self.output_size : y_offset]
		return x, y