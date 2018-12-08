import sqlite3

from modelinfo import ModelInfo

class TensorValveDatabase:
	def __init__(self, path):
		self.connection = sqlite3.connect(path)
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
				name text primary key,
				epochs int,
				minimum_loss real,
				done_training int,
				error text
			)''')

	def get_model_info(self, name):
		parameters = { 'name': '' }
		self.cursor.execute('''
			select
				name,
				epochs,
				minimum_loss,
				done_training,
				error
			from
				model
			where
				name = :name''',
			parameters)
		row = self.cursor.fetchone()
		if row is None:
			return None
		model_info = ModelInfo.from_row(row)
		return model_info

	def set_model_info(self, model_info):
		parameters = {
			'name': model_info.name,
			'epochs': model_info.epochs,
			'minimum_loss': model_info.minimum_loss,
			'done_training': 1 if model_info.done_training else 0,
			'error': model_info.error
		}
		self.cursor.execute('''
			update model set
				epochs = :epochs,
				minimum_loss = :minimum_loss,
				done_training = :done_training,
				error = :error
			where
				name = :name''',
			parameters)

	def get_unfinished_model(self):
		self.cursor.execute('''
			select
				name,
				epochs,
				minimum_loss,
				done_training,
				error
			from
				model
			where
				done_training = 0
				and error is null
			limit 1''')
		row = self.cursor.fetchone()
		if row is None:
			return None
		model_info = ModelInfo.from_row(row)
		return model_info