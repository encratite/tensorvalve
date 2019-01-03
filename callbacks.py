import time

import keyboard
from keras.callbacks import Callback

from profiler import Profiler

class TimeLimitCallback(Callback):
	def __init__(self, time_limit):
		self.time_limit = time_limit
		
	def on_train_begin(self, logs={}):
		self.train_begin_time = time.time()
		
	def on_epoch_end(self, batch, logs={}):
		time_passed = time.time() - self.train_begin_time
		if time_passed >= self.time_limit:
			self.model.stop_training = True

class SaveCallback(Callback):
	def __init__(self, model_path, save_interval = 60):
		self.model_path = model_path
		self.save_interval = save_interval
		self.last_save_time = None
		
	def on_batch_end(self, batch, logs={}):
		if keyboard.is_pressed('q'):
			print('Quitting.')
			self.save()
			self.model.stop_training = True
		
	def on_epoch_end(self, batch, logs={}):
		if self.last_save_time == None or time.time() - self.last_save_time >= self.save_interval:
			self.save()
			self.last_save_time = time.time()

	def save(self):
		profiler = Profiler()
		self.model.save(self.model_path)
		profiler.stop(f'Saved model to "{self.model_path}".')