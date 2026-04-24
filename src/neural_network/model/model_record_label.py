import numpy as np

from src.audio_features.strategy.strategies.strategy_interface import IAFStrategy
from src.audio_features.types import AFTypes
from src.definitions import DURATION, FRAGMENT_LENGTH, labels, labels_colors, labels_annotation, labels_annotation_legend

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from src.neural_network.model.types import ModelTypes
from src.neural_network.model.model_record_evaluator import ModelRecordBaseEvaluator, to_chunks


class ModelRecordColorLabeler(ModelRecordBaseEvaluator):
  def __init__(self,af_strategy: IAFStrategy, af_type: AFTypes, model_type: ModelTypes, model=None, data_set_name: str = 'data_set'):
    super().__init__(af_strategy=af_strategy, af_type=af_type, model_type=model_type, model=model, data_set_name=data_set_name)

  def color_by_label(self, label):
    color = 'black'
    index = labels.index(label)
    if index > -1:
      return labels_colors[index]
    return color


  def _add_legend(self, ax, all_labels, segments_labels):
    legends = []
    legend_labels = []
    for label in all_labels:
      if label in segments_labels:
        if str(label) in labels_annotation:
          if labels_annotation_legend not in legend_labels:
            legends.append(Line2D([0], [0], color=self.color_by_label(label), label=labels_annotation_legend))
            legend_labels.append(labels_annotation_legend)
          continue
        legends.append(Line2D([0], [0], color=self.color_by_label(label), label=label))
        legend_labels.append(str(label).capitalize())
    ax.legend(legends, legend_labels, loc='upper right')

  def _add_annotation(self, ax, segments, colors, segments_labels):
    # Add annotation
    annotations_cords = []
    for i, segment in enumerate(segments):
      color = colors[i]
      try:
        prev_color = colors[i - 1]
        if prev_color != color:
          label = segments_labels[i]
          if labels_annotation.index(label) > -1:
            y = 1
            x = i * len(segment)
            if len(annotations_cords) > 0 and segments_labels[labels_colors.index(prev_color)] != 'noise':
              annotations_cords[-1]['x'] = int(np.mean([annotations_cords[-1]['x'], x]))
            annotations_cords.append({'x': x, 'y': y, 'label': label})
      except ValueError:
        continue

    for annotation in annotations_cords:
      ax.annotate(annotation['label'], (annotation['x'], annotation['y']), color='black', ha='center', va='bottom',
                  fontsize=18)

  def _save_plot(self, file_name: str, export_path: str, segments, colors, segments_labels):
    plt.rcParams.update({
      'font.size': 12,
    })
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.add_collection(LineCollection(segments=segments, colors=colors))
    # Set x and y limits... sadly this is not done automatically for line
    ax.set_xlim(0, len(segments[0]) * len(segments))
    ax.set_ylim(1, -1)

    self._add_annotation(ax, segments, colors, segments_labels)
    self._add_legend(ax, labels, segments_labels)

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
    labels = []
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
      labels.append(line_label)

    self._save_plot(file_name, export_path, segments, colors, labels)

  def label_records(self, export_path: str):
    for file in self.files.get_only_files(self.files_path):
      if file.endswith('.wav'):
        self.label_record(file, export_path)


