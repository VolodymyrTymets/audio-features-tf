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

  def _write_csv(self, file_path, data):
    with open(file_path, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(data)

  def export_model(self, model, label_names, input_shape, target_path: str = None):
    self.loger.log('Saving model...', 'blue')
    export = ModelInstanceFactory(self.model_type).create_model_instance(model, label_names, input_shape)
    tf.saved_model.save(export, target_path)
    self.loger.log('Model is saved to: {}'.format(target_path), 'green')


  def export_history(self, history):
    metrics = history.history
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')

    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, 100 * np.array(metrics['accuracy']), 100 * np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')

    dir_name = self.files.join(self.files.ASSETS_PATH, '__af__', f'{self.af_type.value}_{self.model_type.value}')
    self.files.create_folder(dir_name)
    plt.savefig(self.files.join(dir_name, 'training_history.png'))

  def export_metrics(self, history):
    metrics = history.history
    loss = metrics['loss']
    val_loss = metrics['val_loss']
    accuracy = metrics['accuracy'] * 100
    val_accuracy = metrics['val_accuracy'] * 100
    data = [
             ['loss', 'val_loss', 'accuracy', 'val_accuracy'],
           ] + [[los, val_loss[i], accuracy[i], val_accuracy[i]] for i, los in enumerate(loss)]
    self.loger.log('Saving report...')
    dir_path = self.files.join(self.files.ASSETS_PATH, '__af__', f'{self.af_type.value}_{self.model_type.value}')
    self.files.create_folder(dir_path)


    self._write_csv(self.files.join(dir_path, f'{self.af_type.value}_metics.csv'), data)
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
    data = []
    row_index = None
    col_index = None
    with open(file_path, newline='') as csvfile:
      reader = csv.reader(csvfile)
      for i, row in enumerate(reader):
        header = row if i == 0 else header
        data.append(row)
        for j, col in enumerate(row):
          if af_type.value == header[j] and model_type.value == row[0]:
            row_index = i
            col_index = j
    if row_index is None or col_index is None:
      return
    data[row_index][col_index] = value
    self._write_csv(file_path, data)
