import sys

import scipy.io.wavfile
import tensorflow as tf

from tensorvalve import TensorValve
from profiler import Profiler

def read_wav(path):
	print(f'Loading WAV file from "{path}".')
	rate, data = scipy.io.wavfile.read(path)
	return data

if len(sys.argv) != 5:
	print('Usage:')
	print(f'{sys.argv[0]} <dry training WAV file> <wet training WAV file> <dry validation WAV file> <wet validation WAV file>')
	sys.exit(1)

dry_training_wav_path = sys.argv[1]
wet_training_wav_path = sys.argv[2]

dry_validation_wav_path = sys.argv[3]
wet_validation_wav_path = sys.argv[4]

profiler = Profiler()

dry_training_wav = read_wav(dry_training_wav_path)
wet_training_wav = read_wav(wet_training_wav_path)

dry_validation_wav = read_wav(dry_validation_wav_path)
wet_validation_wav = read_wav(wet_validation_wav_path)

profiler.stop('Done loading WAV files.')

if len(dry_training_wav) != len(wet_training_wav) or len(dry_validation_wav) != len(wet_validation_wav):
	raise Exception('Dry and wet WAVs must be same length.')

layer_types = ('lt', [('lstm', tf.contrib.cudnn_rnn.CudnnLSTM), ('gru', tf.contrib.cudnn_rnn.CudnnGRU)])
time_steps = ('ts', [1, 4, 16, 64])
batch_sizes = ('bs', [1, 4, 16, 64, 256, 512])
frame_counts = ('fc', [32, 64, 128, 256])
dropouts = ('dr', [0.0, 0.25, 0.5])
rnn_bias_initializers = ('bi', [('zero', None), ('xavier', tf.contrib.layers.xavier_initializer())])
activation_functions = ('af', [('elu', tf.nn.elu), ('softsign', tf.nn.softsign), ('tanh', tf.tanh)])
learning_rates = ('lr', [0.001, 0.005, 0.01, 0.05])

time_limit = 15 * 60

# valve = TensorValve(32, 512, 96, 64, save_path)
# valve.train(dry_training_wav, wet_training_wav, dry_validation_wav, wet_validation_wav)