import os
import random
import sys
import time

from wav import read_wav
from profiler import Profiler
from explorer import Explorer
from trainer import Trainer

if len(sys.argv) != 7:
	print('Usage:')
	print(f'python {sys.argv[0]} explore <dry training WAV file> <wet training WAV file> <dry validation WAV file> <wet validation WAV file> <database path>')
	print(f'python {sys.argv[0]} train <dry training WAV file> <wet training WAV file> <dry validation WAV file> <wet validation WAV file> <model path>')
	sys.exit(1)

mode = sys.argv[1]

x_training_path = sys.argv[2]
y_training_path = sys.argv[3]

x_validation_path = sys.argv[4]
y_validation_path = sys.argv[5]

generic_path = sys.argv[6]

use_cpu = True
if use_cpu:
	os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

profiler = Profiler()

x_training_wav = read_wav(x_training_path)
y_training_wav = read_wav(y_training_path)

x_validation_wav = read_wav(x_validation_path)
y_validation_wav = read_wav(y_validation_path)

profiler.stop('Done loading WAV files.')

if len(x_training_wav) != len(y_training_wav) or len(x_validation_wav) != len(y_validation_wav):
	raise Exception('Dry and wet WAVs must be same length.')

if mode == 'explore':
	explorer = Explorer(x_training_wav, y_training_wav, x_validation_wav, y_validation_wav, generic_path)
	explorer.explore()
elif mode == 'train':
	trainer = Trainer(x_training_wav, y_training_wav, x_validation_wav, y_validation_wav, generic_path)
	trainer.train()
else:
	raise Exception('Invalid mode specified.')