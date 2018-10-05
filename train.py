import sys
import wave

def read_wav(path):
	wav = wave.open(path, 'rb')
	sample_width = wav.getsampwidth()
	if sample_width != 3:
		raise Exception('Invalid sample width. Input must be 24-bit.')
	if wav.getnchannels() != 1:
		raise Exception('Invalid channel count. Input must be mono.')
	frame_count = wav.getnframes()
	frames = wav.readframes(frame_count)
	float_samples = []
	for i in range(0, frame_count):
		offset = i * sample_width
		unsigned_value = frames[offset] | (frames[offset + 1] << 8) | (frames[offset + 2] << 16)
		mask = 0x800000
		value = -(unsigned_value & mask) + (unsigned_value & ~mask)
		float_value = value / 2**(8 * sample_width - 1)
		float_samples.append(float_value)
	wav.close()
	output = array('f', float_samples)
	return output

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
