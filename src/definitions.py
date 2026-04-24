ASSETS_PATH = 'assets'

sr = 8000
DURATION = 0.05
FRAGMENT_LENGTH = int(sr / (1 / DURATION))

EPOCHS = 100
SUB_EPOCHS = 20
frame_length = 512
hop_length = frame_length // 4
split_frequency = 2000
n_mels = 64
n_mfcc = 64

distance_points = [5, 10, 15]
labels = ['noise', 'stimulation', 'wrist_extension', 'ecg'] + [str(x) for x in distance_points]
labels_colors = ['blue', 'red', 'green', 'black'] + ['orange'] * len(distance_points)
labels_annotation = [str(x) for x in distance_points]
labels_annotation_legend = 'Distance Points'
sub_sets = ['train', 'test']