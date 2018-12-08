import time

from keras.callbacks import Callback

class TimeLimitCallback(Callback):
	def __init__(self, time_limit):
		self.time_limit = time_limit
		
	def on_train_begin(self, logs={}):
		self.train_begin_time = time.time()
		
	def on_epoch_end(self, batch, logs={}):
		time_passed = time.time() - self.train_begin_time
		if time_passed >= self.time_limit:
			self.model.stop_training = True