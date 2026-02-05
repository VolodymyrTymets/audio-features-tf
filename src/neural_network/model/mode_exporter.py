import csv
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from src.audio_features.types import AFTypes
from src.files import Files
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.export_strategy.model_instance_factory import ModelInstanceFactory
from src.neural_network.model.types import ModelTypes


class MoldeExporter:
  def __init__(self, model_type, af_type):
    self.model_type = model_type
    self.af_type = af_type
    self.files = Files()
    self.loger = Logger('ModelExporter')
    self.export_dir = self.files.join(self.files.ASSETS_PATH, '__af__', f'{self.af_type.value}_{self.model_type.value}')
    self.files.create_folder(self.export_dir)
    self.metrix_file_path = self.files.join(self.export_dir, 'metrics.csv')
    self.training_plot_path = self.files.join(self.export_dir, 'training_history.png')

  def _write_csv(self, file_path, data):
    with open(file_path, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(data)

  def _read_csv(self, file_path):
    data = []
    with open(file_path, newline='') as csvfile:
      reader = csv.reader(csvfile)
      for i, row in enumerate(reader):
        data.append(row)
    return data

  def _read_metrics(self):
   if self.files.is_exist(self.metrix_file_path):
     content = self._read_csv(self.metrix_file_path)
     header = content[0]
     data = content[1:]
     metrics = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
     epoch = []
     for i, row in enumerate(data):
       epoch.append(i + 1)
       for j, col in enumerate(row):
         metrics[header[j]].append(float(row[j]))
     return metrics, epoch
   return None, None


  def export_model(self, model, label_names, input_shape, target_path: str = None):
    self.loger.log('Saving model...', 'blue')
    export = ModelInstanceFactory(self.model_type).create_model_instance(model, label_names, input_shape)
    tf.saved_model.save(export, target_path)
    self.loger.log('Model is saved to: {}'.format(target_path), 'green')


  def export_training_plot(self):
    metrics, epoch = self._read_metrics()
    if metrics is None or epoch is None:
      return

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')

    plt.subplot(1, 2, 2)
    plt.plot(epoch, 100 * np.array(metrics['accuracy']), 100 * np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')

    dir_name = self.files.join(self.files.ASSETS_PATH, '__af__', f'{self.af_type.value}_{self.model_type.value}')
    self.files.create_folder(dir_name)
    plt.savefig(self.training_plot_path)

  def export_metrics(self, metrics: dict):
    loss = metrics['loss']
    val_loss = metrics['val_loss']
    accuracy = metrics['accuracy'] * 100
    val_accuracy = metrics['val_accuracy'] * 100
    data = []
    if self.files.is_exist(self.metrix_file_path):
      data = self._read_csv(self.metrix_file_path)
    else:
      data.append(['loss', 'val_loss', 'accuracy', 'val_accuracy'])
    data += [[los, val_loss[i], accuracy[i], val_accuracy[i]] for i, los in enumerate(loss)]
    self.loger.log('Saving report...')
    self._write_csv(self.metrix_file_path, data)
    self.loger.log('Report saved!', 'green')

  def init_evaluation_export(self, file_path):
    if self.files.is_exist(file_path):
      return
    self.loger.log('Initializing evaluation report...')
    data = [["name"] + [_af_type.value for _af_type in AFTypes]]
    for _model_type in ModelTypes:
      col = [_model_type.value]
      for _af_type in AFTypes:
          col.append('0')
      data.append(col)

    self._write_csv(file_path, data)

  def export_evaluation(self, value:int, af_type: AFTypes, model_type: ModelTypes, file_name: str = 'evaluation'):
    file_path = self.files.join(self.files.ASSETS_PATH, '__af__', f'{file_name}.csv')
    self.init_evaluation_export(file_path)
    data = self._read_csv(file_path)
    row_index = None
    col_index = None
    header = None
    for i, row in enumerate(data):
      header = row if i == 0 else header
      for j, col in enumerate(row):
        if af_type.value == header[j] and model_type.value == row[0]:
          row_index = i
          col_index = j
    if row_index is None or col_index is None:
      return
    data[row_index][col_index] = value
    self._write_csv(file_path, data)
