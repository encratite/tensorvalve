import os

from keras import Sequential
from keras.layers import LSTM, GRU, Dense, ELU
from keras.models import load_model
from keras.optimizers import Adam

from loss import root_mean_squared_error
from callbacks import SaveCallback
from profiler import Profiler
from generator import SlidingWindowGenerator

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
		rnn_units = 128
		rnn_timesteps = 128
		rnn_features = 32

		output_size = 1

		batch_size = 512
		epochs = 10000000

		input_shape = (rnn_timesteps, rnn_features)

		if os.path.isfile(self.model_path):
			profiler = Profiler()
			model = load_model(self.model_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
			profiler.stop(f'Loaded model from "{self.model_path}".')
		else:
			model = Sequential()
			model.add(rnn_type(rnn_units, dropout=rnn_dropout, return_sequences=False, input_shape=input_shape))
			model.add(Dense(output_size))
			model.add(ELU())
			optimizer = Adam(lr=0.001)
			model.compile(optimizer=optimizer, loss=root_mean_squared_error)

		training_generator = SlidingWindowGenerator(self.x_training_wav, self.y_training_wav, input_shape, output_size, batch_size)
		validation_generator = SlidingWindowGenerator(self.x_validation_wav, self.y_validation_wav, input_shape, output_size, batch_size)
		save_callback = SaveCallback(self.model_path)
		history = model.fit_generator(generator=training_generator, epochs=epochs, verbose=1, validation_data=validation_generator, callbacks=[save_callback])