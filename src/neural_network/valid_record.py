import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.audio_features.types import AFTypes
from src.definitions import DURATION, FRAGMENT_LENGTH
from src.files import Files
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

files = Files()

def to_chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    yield lst[i:i + n]

def get_chunk_label_by_model(model, wave):
  x = tf.convert_to_tensor(wave, dtype=tf.float32)
  waveform = x[tf.newaxis, ...]
  result = model(tf.constant(waveform))
  label_names = np.array(result['label_names'])
  prediction = tf.nn.softmax(result['predictions']).numpy()[0]
  max_value = max(prediction)
  i, = np.where(prediction == max_value)
  wave_label = label_names[i]
  return wave_label


def valid_record(af_type: AFTypes, show_plot=False):

  model_dir = files.join(files.ASSETS_PATH, 'models', f'm_{DURATION}_{af_type.value}')
  model = tf.saved_model.load(model_dir)
  files_path = files.join(files.ASSETS_PATH, 'test', 'records')
  for file in files.get_only_files(files_path):


    file_path = os.path.join(files_path, file)
    waveform, sr = librosa.load(file_path)
    chunks = [x for x in to_chunks(waveform, int(FRAGMENT_LENGTH))]
    chunks_n = to_chunks(waveform, int(FRAGMENT_LENGTH))
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
        line_label = get_chunk_label_by_model(model, chunk_n)
      except:
        line_label = 'noise'
      color = 'red' if 'stimulation' in str(line_label) else 'blue'
      color = 'green' if 'breath' in str(line_label) else color
      linecolors.append(color)

    print('Labe with duration {}: {}'.format(DURATION, file_path))
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