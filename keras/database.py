import sqlite3

class ModelDatabase:
	def __init__(self, path):
		self.connection = sqlite3.connect(path, isolation_level=None)
		self.cursor = self.connection.cursor()
		self.create_tables()

	def __enter__(self):
		return self

	def __exit__(self):
		self.connection.close()

	def create_tables(self):
		self.cursor.execute('''
			create table if not exists model
			(
				mode text,
				rnn_type text,
				rnn_dropout real,
				rnn_units int,
				rnn_timesteps int,
				batch_size int,
				time_limit int,
				loss real,
				val_loss real
			)''')

	def model_info_exists(self, options):
		self.cursor.execute('''
			select
				count(*)
			from
				model
			where
				mode = :mode
				and rnn_type = :rnn_type
				and rnn_dropout = :rnn_dropout
				and rnn_units = :rnn_units
				and rnn_timesteps = :rnn_timesteps
				and batch_size = :batch_size
				and time_limit = :time_limit''',
			options)
		row = self.cursor.fetchone()
		count = row[0]
		return count > 0

	def save_model_info(self, values):
		self.cursor.execute('''
			insert into model
			(
				mode,
				rnn_type,
				rnn_dropout,
				rnn_units,
				rnn_timesteps,
				batch_size,
				time_limit,
				loss,
				val_loss
			)
			values
			(
				:mode,
				:rnn_type,
				:rnn_dropout,
				:rnn_units,
				:rnn_timesteps,
				:batch_size,
				:time_limit,
				:loss,
				:val_loss
			)''',
		values)