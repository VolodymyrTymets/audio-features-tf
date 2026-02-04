import os
import numpy as np
import tensorflow as tf

from src.audio_features.types import AFTypes
from src.definitions import DURATION, FRAGMENT_LENGTH, frame_length, hop_length, sr as SR
from src.files import Files
from src.wav_files import WavFiles
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from src.neural_network.utils import label_by_model

from src.audio_features.strategy.af_strategy_factory import AFStrategyFactory
from src.neural_network.model.types import ModelTypes

files = Files()
wav_files = WavFiles()

def pad_array(arr, length):
  if len(arr) < length:
    return np.pad(arr, (0, length - len(arr)), 'constant', constant_values=0)
  return arr

def to_chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    yield lst[i:i + n]

def valid_record(af_type: AFTypes, model_type: ModelTypes, name = ''):
  model_dir = files.join(files.ASSETS_PATH, 'models', f'm_{DURATION}_{af_type.value}_{model_type.value}')
  model = tf.saved_model.load(model_dir)
  files_path = files.join(files.ASSETS_PATH, 'test', 'records')
  for file in files.get_only_files(files_path):
    if name != '' and name not in file:
      continue
    file_path = os.path.join(files_path, file)
    waveform, _ = wav_files.read(file_path)
    chunks = [x for x in to_chunks(waveform, int(FRAGMENT_LENGTH))]
    chunks_n = to_chunks(waveform, int(FRAGMENT_LENGTH))
    strategy = AFStrategyFactory(sr=SR, frame_length=frame_length, hop_length=hop_length).create_strategy(af_type)
    # Form segments for collection of lines
    segments = []
    linecolors = []
    x = 0
    for i, chunk_n in enumerate(chunks_n):
      lineN = []
      lin_y = chunks[i]
      for i, y in enumerate(lin_y):
        lineN.append((x, y))
        x = x + 1
      segments.append(lineN)
      try:
        signal = pad_array(chunk_n, FRAGMENT_LENGTH)
        af = strategy.get_audio_feature(signal=signal)

        line_label, _ = label_by_model(model, af)
      except Exception as e:
        line_label = 'noise'
        print(e)
      color = 'red' if 'stimulation' in str(line_label) else 'blue'
      color = 'green' if 'breath' in str(line_label) else color
      linecolors.append(color)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 2))
    line_collection = LineCollection(segments=segments, colors=linecolors)
    # Add a collection of lines
    ax.add_collection(line_collection)

    # Set x and y limits... sadly this is not done automatically for line
    # collections
    ax.set_xlim(0, len(waveform))
    ax.set_ylim(1, -1)
    ax.legend(
      [Line2D([0, 1], [0, 1], color='blue'), Line2D([0, 1], [0, 1], color='green'), Line2D([0, 1], [0, 1], color='red')],
      ['Noise', 'Breath', 'Stimulation'])
    # plt.show()
    dir_name = files.join(files.ASSETS_PATH, '__af__', af_type.value, 'records')
    files.create_folder(dir_name)
    plt.savefig(files.join(dir_name, file.replace('.wav', '.png')))
    print(f'[{DURATION}_{af_type.value}_{model_type.value}] Labeled: {file_path}')