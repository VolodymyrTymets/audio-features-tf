ASSETS_PATH = 'assets'

sr = 44100
DURATION = 0.2
FRAGMENT_LENGTH = int(sr / (1 / DURATION))

EPOCHS = 1000
SUB_EPOCHS = 100
frame_length = 512
hop_length = frame_length // 4
split_frequency = 2000
n_mels = 64
n_mfcc = 64

# labels = ['noise', 'breath', 'stimulation']
labels = ['noise', 'breath', 'rln'] + [str(i) for i in range(1, 11)]
labels_colors = ['blue', 'green', 'red'] + [(168/255, 50/255, (10*i)/255) for i in range(1, 11)]
labels_annotation = [str(i) for i in range(1, 11)]
sub_sets = ['train', 'test']