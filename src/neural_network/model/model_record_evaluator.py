import os
import numpy as np
import tensorflow as tf
import json
import time
from contextlib import contextmanager

from numpy.core.records import record

from src.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.audio_features.types import AFTypes
from src.definitions import DURATION, FRAGMENT_LENGTH, labels, labels_colors, labels_annotation
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
    self._get_audio_feature_times = []
    self.label_by_model_times = []

    if model is None:
      raise ValueError("ModelRecordBaseEvaluator Model not set")
    else:
      self.model = model
    self.files_path = self.files.join(self.files.ASSETS_PATH, 'test', 'records')

  @contextmanager
  def _mesure_duration(self, results: list[float]):
    start = time.perf_counter()
    try:
      yield
    finally:
      elapsed_ms = (time.perf_counter() - start) * 1000.0
      results.append(elapsed_ms)

  def _label_by_model(self, signal: np.ndarray):
    try:
      signal = pad_array(signal, FRAGMENT_LENGTH)
      with self._mesure_duration(self._get_audio_feature_times):
        af = self.strategy.get_audio_feature(signal=signal)
        with self._mesure_duration(self.label_by_model_times):
          label = label_by_model(self.model, af)
          return label
    except Exception as e:
      line_label = 'noise'
      self.loger.error(e)
      return line_label, 0

  def get_af_times(self):
    return self._get_audio_feature_times

  def get_label_by_model_times(self):
    return self.label_by_model_times


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
    if not annotations.keys():
      return 0
    model_annotation = []

    timestamp = 0
    chunks = [x for x in to_chunks(waveform, int(FRAGMENT_LENGTH))]
    rate_per_chunk = 100 / len(chunks)
    evaluate_rate = 0

    for chunk in chunks:
      duration = 1 / sr * len(chunk)
      start = timestamp
      end = timestamp + duration
      line_label, prediction = self._label_by_model(chunk)
      model_annotation.append([start, end, line_label])
      chunk_annotation_label = ''
      for label in labels:
        chunk_annotation = annotations.get(label, {})

        if chunk_annotation:
          is_chunk_in_annotation = self.is_in_timestamp(start, end, chunk_annotation)
          if is_chunk_in_annotation:
            chunk_annotation_label = label
            break
        else:
          chunk_annotation_label = label
      chunk_evaluate_rate = rate_per_chunk if line_label == chunk_annotation_label else 0
      timestamp += duration
      evaluate_rate += chunk_evaluate_rate
    # model_annotation_str = '\n'.join([f'{line_label}:[{start},{end}]' for start, end, line_label in model_annotation])
    self.loger.log(f'Record labels: {file_name}:')
    # self.loger.log(model_annotation_str)
    self.loger.log(f'Evaluate rate: {evaluate_rate} %')
    self.loger.log(f'')
    return evaluate_rate

  def time_record(self, file_name: str,   shift = 0, log=False):
    file_path = self.files.join(self.files_path, file_name)
    waveform, _ = self.wav_files.read(file_path)
    count = 0
    start = time.perf_counter()
    for chunk in to_chunks(waveform, int(FRAGMENT_LENGTH)):
      self._label_by_model(chunk)
      count += 1
    elapsed_ms = (time.perf_counter() - start)

    af_time = np.mean(self.get_af_times())
    label_time = np.mean(self.get_label_by_model_times())
    record_time = elapsed_ms + ((shift * count) / 1000)
    if log == True:
      self.loger.log(f'__________{self.af_type.value}__________')
      self.loger.log(f'get {self.af_type.value} time: {af_time + shift} ms')
      self.loger.log(f'label {self.af_type.value} time: {label_time + shift} ms')
      self.loger.log(f'label record : {record_time} s')
      self.loger.log(f'________________________________________')
    return af_time + shift, label_time + shift, record_time

  def evaluate_records(self):
    test_record_acc = []
    for file in self.files.get_only_files(self.files_path):
      if file.endswith('.wav'):
        test_record_acc.append(self.evaluate_record(file))
    return np.mean(test_record_acc)

  def time_records(self, shift = 0):
    af_time_acc = []
    fragment_time_acc = []
    record_time_acc = []
    for file in self.files.get_only_files(self.files_path):
      if file.endswith('.wav'):
        af_time, fragment_time, record_time = self.time_record(file, shift)
        af_time_acc.append(af_time)
        fragment_time_acc.append(fragment_time)
        record_time_acc.append(record_time)
    return np.mean(af_time_acc), np.mean(fragment_time_acc), np.mean(record_time_acc)



class ModelRecordColorLabeler(ModelRecordBaseEvaluator):
  def __init__(self,af_strategy: IAFStrategy, af_type: AFTypes, model_type: ModelTypes, model=None):
    super().__init__(af_strategy=af_strategy, af_type=af_type, model_type=model_type, model=model)

  def color_by_label(self, label):
    color = 'black'
    index = labels.index(label)
    if index > -1:
      return labels_colors[index]
    return color

  def _save_plot(self, file_name: str, export_path: str, segments, colors):
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.add_collection(LineCollection(segments=segments, colors=colors))


    # Add annotation
    annotatations_cords = []
    for i, segment in enumerate(segments):
      color = colors[i]
      try:
        prev_color = colors[i - 1]
        if prev_color != color:
          label = labels[labels_colors.index(color)]
          if labels_annotation.index(label) > -1:
            y = 1
            x = i * len(segment)
            if len(annotatations_cords) > 0 and labels[labels_colors.index(prev_color)] != 'noise':
              annotatations_cords[-1]['x'] = int(np.mean([annotatations_cords[-1]['x'], x]))
            annotatations_cords.append({ 'x': x, 'y': y, 'label': label })
      except ValueError:
        continue

    for annotation in annotatations_cords:
      ax.annotate(annotation['label'], (annotation['x'], annotation['y']), color='black', ha='center', va='bottom', fontsize=18)
    # Set x and y limits... sadly this is not done automatically for line
    ax.set_xlim(0, len(segments[0]) * len(segments))
    ax.set_ylim(1, -1)
    legends = []
    legend_labels = []
    for label in labels:
      try:
        labels_annotation.index(label)
        continue
      except ValueError:
        legends.append(Line2D([0], [0], color=self.color_by_label(label), label=label))
        legend_labels.append(label.capitalize())
    ax.legend(legends, legend_labels, loc='upper right',)
    dir_name = self.files.join(export_path, 'records')
    self.files.create_folder(dir_name)
    plt.savefig(self.files.join(dir_name, file_name.replace('.wav', '.png')))
    print(f'[{DURATION}_{self.af_type.value}_{self.model_type.value}] Labeled: {file_name}')

  def label_record(self, file_name: str, export_path: str):
    file_path = self.files.join(self.files_path, file_name)
    waveform, _ = self.wav_files.read(file_path)

    # Form segments for collection of lines
    segments = []
    colors = []
    x = 0
    for i, chunk in enumerate(to_chunks(waveform, int(FRAGMENT_LENGTH))):
      segment = []
      for y in chunk:
        segment.append((x, y))
        x = x + 1
      segments.append(segment)
      line_label, _ = self._label_by_model(chunk)

      color = self.color_by_label(line_label)
      # color = color if color != labels_colors[0] else (labels_colors[0] if i % 2 == 0 else 'black' )
      colors.append(color)

    self._save_plot(file_name, export_path, segments, colors)

  def label_records(self, export_path: str):
    for file in self.files.get_only_files(self.files_path):
      if file.endswith('.wav'):
        self.label_record(file, export_path)


