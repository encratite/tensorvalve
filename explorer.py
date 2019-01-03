from keras import Sequential
from keras.layers import LSTM, GRU

from loss import root_mean_squared_error
from wav import reshape_wav
from profiler import Profiler
from callbacks import TimeLimitCallback
from database import ModelDatabase

class Explorer:
	def __init__(self, x_training_wav, y_training_wav, x_validation_wav, y_validation_wav, database_path):
		self.x_training_wav = x_training_wav
		self.y_training_wav = y_training_wav
		self.x_validation_wav = x_validation_wav
		self.y_validation_wav = y_validation_wav
		self.database_path = database_path

	def explore(self):
		option_values = {
			'mode': [mode],
			'rnn_type': ['LSTM', 'GRU'],
			'rnn_dropout': [0, 0.25, 0.5],
			'rnn_units': [32, 64, 128, 256],
			'rnn_timesteps': [16, 32, 64, 128, 256],
			'batch_size': [16, 32, 64, 128, 256],
			'time_limit': [15 * 60]
		}

		with ModelDatabase(self.database_path) as database:
			while True:
				try:
					print('Options:')
					options = {}
					for key in option_values:
						values = option_values[key]
						value = random.choice(values)
						options[key] = value
						print(f'{key} = {value}')
					if database.model_info_exists(options):
						print('Skipping permutation.')
						time.sleep(1)
						continue
					loss, val_loss = train(options)
					values = dict(options)
					values['loss'] = loss
					values['val_loss'] = val_loss
					database.save_model_info(values)
				except Exception as error:
					print(error)
					time.sleep(5)
					continue

	def train(self, options):
		rnn_types = {
			'LSTM': LSTM,
			'GRU': GRU
		}
		rnn_type_key = options['rnn_type']
		rnn_type = rnn_types[rnn_type_key]
	
		rnn_dropout = options['rnn_dropout']
		rnn_units = options['rnn_units']
		rnn_timesteps = options['rnn_timesteps']
		rnn_features = rnn_units

		batch_size = options['batch_size']
		epochs = 1000000
		time_limit = options['time_limit']

		profiler = Profiler()

		input_shape = (rnn_timesteps, rnn_features)
	
		x_training = reshape_wav(self.x_training_wav, input_shape)
		y_training = reshape_wav(self.y_training_wav, input_shape)
	
		x_validation = reshape_wav(self.x_validation_wav, input_shape)
		y_validation = reshape_wav(self.y_validation_wav, input_shape)
		
		model = Sequential()
		model.add(rnn_type(rnn_units, dropout=rnn_dropout, return_sequences=True, input_shape=input_shape))
		model.compile('adam', loss=root_mean_squared_error)

		time_limit_callback = TimeLimitCallback(time_limit)
		history = model.fit(x_training, y_training, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_validation, y_validation), callbacks=[time_limit_callback])

		profiler.stop('Done training.')

		loss = history.history['loss'][-1]
		val_loss = history.history['val_loss'][-1]

		return (loss, val_loss)
