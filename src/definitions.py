ASSETS_PATH = 'assets'

sr = 22050
DURATION = 0.05
FRAGMENT_LENGTH = int(sr / (1 / DURATION))

EPOCHS = 100
SUB_EPOCHS = 20
frame_length = 512
hop_length = frame_length // 4
split_frequency = 2000
n_mels = 64
n_mfcc = 64

labels = ['noise', 'stimulation']
labels_colors = ['blue', 'red']
labels_annotation = []
sub_sets = ['train', 'test']