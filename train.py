import sys

import scipy.io.wavfile

def read_wav(path):
	rate, data = scipy.io.wavfile.read(path)
	return data

if len(sys.argv) != 3:
	print('Usage:')
	print(f'{sys.argv[0]} <dry.wav> <wet.wav>')
	sys.exit(1)

dry_path = sys.argv[1]
wet_path = sys.argv[2]
dry_wav = read_wav(dry_path)
wet_wav = read_wav(wet_path)
if len(dry_wav) < len(wet_wav):
	raise Exception('Wet recording must be at least as long as dry recording.')
