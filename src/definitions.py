ASSETS_PATH = 'assets'

sr = 44100
DURATION = 0.2
FRAGMENT_LENGTH = int(sr / (1 / DURATION))

EPOCHS = 10

frame_length = 512
hop_length = frame_length // 4