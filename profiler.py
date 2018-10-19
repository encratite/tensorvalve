import time

class Profiler:
	def __init__(self):
		self.time = time.perf_counter()

	def stop(self, message):
		new_time = time.perf_counter()
		difference = new_time - self.time
		self.time = new_time
		print(f'{message}: {difference * 1000:2f} ms')