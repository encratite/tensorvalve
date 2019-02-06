import os
import sys
from math import sqrt

import scipy.io.wavfile

from wav import read_wav

def get_root_mean_square(x, y, y_offset):
	sum = 0
	for i in range(0, len(x)):
		sum += (x[i] - y[y_offset + i])**2
	root_mean_square = sqrt(sum / len(x))
	return root_mean_square

def translate_block(input_block, dry_model, wet_model):
	best_offset = None
	best_rms = None
	block_size = len(input_block)
	final_offset = len(dry_model) - block_size
	for offset in range(1, final_offset):
		if offset % 1000 == 0:
			print(f'Progress: {float(offset) / final_offset}')
		rms = get_root_mean_square(input_block, dry_model, offset)
		if best_rms is None or rms < best_rms:
			best_offset = offset
			best_rms = rms
			print(f'Best RMS: {best_rms} (at {best_offset})')
	translated_block = wet_model[best_offset : best_offset + block_size]
	return translated_block

if len(sys.argv) != 6:
	print('Usage:')
	print(f'python {sys.argv[0]} <dry training WAV file> <wet training WAV file> <dry validation WAV file> <wet validation WAV file> <output WAV file>')
	sys.exit(1)
	
dry_training_wav_path = sys.argv[1]
wet_training_wav_path = sys.argv[2]
dry_validation_wav_path = sys.argv[3]
wet_validation_wav_path = sys.argv[4]
output_wav_path = sys.argv[5]

dry_training_wav = read_wav(dry_training_wav_path)
wet_training_wav = read_wav(wet_training_wav_path)
dry_validation_wav = read_wav(dry_validation_wav_path)
wet_validation_wav = read_wav(wet_validation_wav_path)

training_offset = 0
for i in range(0, len(dry_training_wav)):
	if dry_training_wav[i] != 0:
		training_offset = i
		break
dry_training_wav = dry_training_wav[training_offset : ]
wet_training_wav = wet_training_wav[training_offset : ]

block_size = 128

offset = 21651
input_block = dry_validation_wav[offset : offset + block_size]
translated_block = translate_block(input_block, dry_training_wav, wet_training_wav)
final_rms = get_root_mean_square(translated_block, wet_validation_wav, offset)
print(f'Final RMS: {final_rms}')