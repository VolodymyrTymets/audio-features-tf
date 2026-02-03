
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.definitions import FRAGMENT_LENGTH
from src.files import Files
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.preporcess_strategy.preprocess_strategy_interface import IModelPreprocessStrategy
from src.audio_features.strategy.strategies.strategy_interface import IAFStrategy


class CNNModelPreprocessStrategy(IModelPreprocessStrategy):
  def __init__(self, af_strategy: IAFStrategy):
    self.files = Files()
    self.loger = Logger('CNNModelPreprocessStrategy')
    self.af_strategy = af_strategy
    self.shape = self._calculate_shape()

  def _calculate_shape(self):
    signal = np.zeros(FRAGMENT_LENGTH)
    af = self.af_strategy.get_audio_feature(signal)
    return af.shape

  def get_bunch_audio_feature(self, waveform):
    bunch = waveform.numpy()
    return np.array([self.af_strategy.get_audio_feature(signal) for signal in bunch])

  def save_bunch_audio_feature(self, af, labels, label_names):
    bunch = af.numpy()
    first = bunch[0]
    first_label_index = labels.numpy()[0]
    label = label_names.numpy()[first_label_index].decode("utf-8")
    self.af_strategy.save_audio_feature(first, label=label)
    return af

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
  def reshape(self, i):
    w, h = self.shape
    return tf.reshape(i, (-1, w, h, 1))

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
  def get_audio_feature(self, i):
    return tf.py_function(self.get_bunch_audio_feature, [i], tf.float32)

  @tf.function
  def save_audio_feature(self, i, labels, label_names):
    tf.py_function(self.save_bunch_audio_feature, [i, labels, label_names], tf.float32)
    return i

  def preprocess(self, data_set_path: str, save_af: bool = False):
    # Form data storage
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
      directory=self.files.join(data_set_path, 'train'),
      batch_size=32,
      validation_split=0.2,
      seed=0,
      output_sequence_length=FRAGMENT_LENGTH,
      subset='both')
    test_ds = tf.keras.utils.audio_dataset_from_directory(
      directory=self.files.join(data_set_path, 'test'),
      batch_size=32,
      seed=0,
      output_sequence_length=FRAGMENT_LENGTH)

    label_names = np.array(train_ds.class_names)

    # Prepare data - wave to audio feature
    train_ds = train_ds.map(lambda audio, label: (tf.squeeze(audio, axis=-1), label), tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda audio, label: (tf.squeeze(audio, axis=-1), label), tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda audio, label: (tf.squeeze(audio, axis=-1), label), tf.data.AUTOTUNE)

    train_ds = train_ds.map(lambda i, label: (self.get_audio_feature(i), label), tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda i, label: (self.get_audio_feature(i), label), tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda i, label: (self.get_audio_feature(i), label), tf.data.AUTOTUNE)

    if save_af:
      train_ds = train_ds.map(lambda af, l: (self.save_audio_feature(af, l, label_names), l), tf.data.AUTOTUNE)

    train_ds = train_ds.map(lambda i, l: (self.reshape(i), l), tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda i, l: (self.reshape(i), l), tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda i, l: (self.reshape(i), l), tf.data.AUTOTUNE)

    # Training model
    train_ds = train_ds.cache().shuffle(
      10000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, label_names
