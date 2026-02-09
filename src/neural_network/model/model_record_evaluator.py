import os
import numpy as np
import tensorflow as tf
import json

from src.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.audio_features.types import AFTypes
from src.definitions import DURATION, FRAGMENT_LENGTH, labels
from src.files import Files
from src.logger.logger_service import Logger
from src.wav_files import WavFiles
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from src.neural_network.utils import label_by_model

from src.neural_network.model.types import ModelTypes

def pad_array(arr, length):
  if len(arr) < length:
    return np.pad(arr, (0, length - len(arr)), 'constant', constant_values=0)
  return arr

def to_chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    yield lst[i:i + n]


class ModelRecordBaseEvaluator:
  def __init__(self,af_strategy: IAFStrategy, af_type: AFTypes, model_type: ModelTypes, model=None):
    self.files = Files()
    self.wav_files = WavFiles()
    self.af_type = af_type
    self.model_type = model_type
    self.strategy = af_strategy
    self.loger = Logger('ModelRecordEvaluator')

    if model is None:
      self.loger.log('Loading model...')
      model_dir = self.files.join(self.files.ASSETS_PATH, 'models', f'm_{DURATION}_{self.af_type.value}_{self.model_type.value}')
      self.model = tf.saved_model.load(model_dir)
    else:
      self.model = model
    self.files_path = self.files.join(self.files.ASSETS_PATH, 'test', 'records')

  def _label_by_model(self, signal: np.ndarray):
    try:
      signal = pad_array(signal, FRAGMENT_LENGTH)
      af = self.strategy.get_audio_feature(signal=signal)
      return label_by_model(self.model, af)
    except Exception as e:
      line_label = 'noise'
      self.loger.error(e)
      return line_label, 0

class ModelRecordEvaluator(ModelRecordBaseEvaluator):
  def __init__(self, af_strategy: IAFStrategy, af_type: AFTypes, model_type: ModelTypes, model=None):
    super().__init__(af_strategy=af_strategy, af_type=af_type, model_type=model_type, model=model)

  def _load_annotation(self, file_name: str):
    annotation_file = file_name.replace('.wav', f'.annotation.{DURATION}.json')
    annotation_path = self.files.join(self.files_path, annotation_file)
    annotations = {}
    if os.path.exists(annotation_path):
      with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    return annotations

  def is_in_timestamp(self, start: float, end: float, timestamps: list[list[float]]):
    is_in = False
    for timestamp in timestamps:
      if start >= timestamp[0] and end <= timestamp[1]:
        is_in = True
        break
    return is_in

  def evaluate_record(self, file_name: str):
    file_path = self.files.join(self.files_path, file_name)
    waveform, sr = self.wav_files.read(file_path)
    annotations = self._load_annotation(file_name)

    timestamp = 0
    chunks = [x for x in to_chunks(waveform, int(FRAGMENT_LENGTH))]
    rate_per_chunk = 100 / len(chunks)
    evaluate_rate = 0

    for chunk in chunks:
      duration = 1 / sr * len(chunk)
      start = timestamp
      end = timestamp + duration
      line_label, prediction = self._label_by_model(chunk)
      chunk_annotation_label = ''
      for label in labels:
        chunk_annotation = annotations.get(label, {})
        if chunk_annotation:
          is_chunk_in_annotation = self.is_in_timestamp(start, end, chunk_annotation)
          if is_chunk_in_annotation:
            chunk_annotation_label = label
        else:
          chunk_annotation_label = label

      chunk_evaluate_rate = rate_per_chunk if line_label == chunk_annotation_label else 0
      timestamp += duration
      evaluate_rate += chunk_evaluate_rate

    return evaluate_rate

class ModelRecordColorLabeler(ModelRecordBaseEvaluator):
  def __init__(self,af_strategy: IAFStrategy, af_type: AFTypes, model_type: ModelTypes, model=None):
    super().__init__(af_strategy=af_strategy, af_type=af_type, model_type=model_type, model=model)

  def color_by_label(self, label):
    color = 'blue'
    if label == 'breath':
      color = 'green'
    elif label == 'stimulation':
      color = 'red'
    return color

  def _save_plot(self, file_name, segments, colors, ):
    fig, ax = plt.subplots(figsize=(12, 2))
    line_collection = LineCollection(segments=segments, colors=colors)
    # Add a collection of lines
    ax.add_collection(line_collection)

    # Set x and y limits... sadly this is not done automatically for line
    ax.set_xlim(0, len(segments[0]) * len(segments))
    ax.set_ylim(1, -1)
    ax.legend(
      [Line2D([0, 1], [0, 1], color='blue'), Line2D([0, 1], [0, 1], color='green'),
       Line2D([0, 1], [0, 1], color='red')],
      ['Noise', 'Breath', 'Stimulation'])
    dir_name = self.files.join(self.files.ASSETS_PATH, '__af__', f'{self.af_type.value}_{self.model_type.value}',
                               'records')
    self.files.create_folder(dir_name)
    plt.savefig(self.files.join(dir_name, file_name.replace('.wav', '.png')))
    print(f'[{DURATION}_{self.af_type.value}_{self.model_type.value}] Labeled: {file_name}')

  def label_record(self, file_name: str):
    file_path = self.files.join(self.files_path, file_name)
    waveform, _ = self.wav_files.read(file_path)

    # Form segments for collection of lines
    segments = []
    colors = []
    x = 0
    for chunk in to_chunks(waveform, int(FRAGMENT_LENGTH)):
      segment = []
      for y in chunk:
        segment.append((x, y))
        x = x + 1
      segments.append(segment)
      line_label, _ = self._label_by_model(chunk)
      colors.append(self.color_by_label(line_label))
    self._save_plot(file_name, segments, colors)

  def label_records(self):
    for file in self.files.get_only_files(self.files_path):
      if file.endswith('.wav'):
        self.label_record(file)


