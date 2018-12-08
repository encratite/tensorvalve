class ModelInfo:
	def __init__(self, name, epochs, minimum_loss, done_training, error):
		self.name = name
		self.epochs = epochs
		self.minimum_loss = minimum_loss
		self.done_training = done_training
		self.error = error

	def from_row(row):
		model_info = ModelInfo(
			row['name'],
			row['epochs'],
			row['minimum_loss'],
			row['done_training'] == 1,
			row['error']
		)
		return model_info