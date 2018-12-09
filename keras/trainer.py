import os

from keras import Sequential
from keras.layers import LSTM, GRU
from keras.models import load_model

from loss import root_mean_squared_error
from wav import reshape_wav
from callbacks import SaveCallback
from profiler import Profiler

class Trainer:
	def __init__(self, x_training_wav, y_training_wav, x_validation_wav, y_validation_wav, model_path):
		self.x_training_wav = x_training_wav
		self.y_training_wav = y_training_wav
		self.x_validation_wav = x_validation_wav
		self.y_validation_wav = y_validation_wav
		self.model_path = model_path

	def train(self):
		rnn_type = LSTM
	
		rnn_dropout = 0.0
		rnn_units = 32
		rnn_timesteps = 64
		rnn_features = rnn_units

		batch_size = 32
		epochs = 10000000

		input_shape = (rnn_timesteps, rnn_features)
	
		x_training = reshape_wav(self.x_training_wav, input_shape)
		y_training = reshape_wav(self.y_training_wav, input_shape)
	
		x_validation = reshape_wav(self.x_validation_wav, input_shape)
		y_validation = reshape_wav(self.y_validation_wav, input_shape)

		if os.path.isfile(self.model_path):
			profiler = Profiler()
			model = load_model(self.model_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
			profiler.stop(f'Loaded model from "{self.model_path}".')
		else:
			model = Sequential()
			model.add(rnn_type(rnn_units, dropout=rnn_dropout, return_sequences=True, input_shape=input_shape))
			model.compile('adam', loss=root_mean_squared_error)

		save_callback = SaveCallback(self.model_path)
		history = model.fit(x_training, y_training, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_validation, y_validation), callbacks=[save_callback])