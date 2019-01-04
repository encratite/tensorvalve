from keras.utils import Sequence

class SlidingWindowGenerator(Sequence):
	def __init__(self, x_wav, y_wav, window_size, output_size):
		self.x_wav = x_wav
		self.y_wav = y_wav
		self.window_size = window_size
		self.output_size = output_size

	def __len__(self):
		length = len(self.y_wav) - self.window_size - self.output_size
		return length

	def __getitem__(self, index):
		y_offset = index + self.window_size
		x = self.x_wav[index : y_offset]
		y = self.y_wav[y_offset : y_offset + self.output_size]
		return x, y