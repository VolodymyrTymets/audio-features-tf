import csv
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from src.definitions import DURATION
from src.files import Files
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.export_strategy.model_instance_factory import ModelInstanceFactory


class MoldeExporter:
  def __init__(self, model_type, af_type):
    self.model_type = model_type
    self.af_type = af_type
    self.files = Files()
    self.loger = Logger('ModelExporter')

  def export_model(self, model, label_names, input_shape):
    self.loger.log('Saving model...', 'blue')
    model_dir = self.files.join(self.files.ASSETS_PATH, 'models',
                                'm_{}_{}_{}'.format(DURATION, self.af_type.value, self.model_type.value))
    export = ModelInstanceFactory(self.model_type).create_model_instance(model, label_names, input_shape)
    tf.saved_model.save(export, model_dir)
    self.loger.log('Model is saved to: {}'.format(model_dir), 'green')

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
    print('Saving report...')
    dir_path = self.files.join(self.files.ASSETS_PATH, '__af__', f'{self.af_type.value}_{self.model_type.value}')
    self.files.create_folder(dir_path)

    with open(self.files.join(dir_path, f'{self.af_type.value}_metics.csv'), 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(data)
    print('Report saved!')
