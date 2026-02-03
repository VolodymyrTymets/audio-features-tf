ASSETS_PATH = 'assets'

sr = 44100
DURATION = 0.2
FRAGMENT_LENGTH = int(sr / (1 / DURATION))

EPOCHS = 20

frame_length = 512
hop_length = frame_length // 4
split_frequency = 2000
n_mels = 64
n_mfcc = 64

labels = ['noise', 'breath', 'stimulation']
sub_sets = ['train', 'test']