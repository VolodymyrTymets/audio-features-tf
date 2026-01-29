
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.audio_features.types import AFTypes
from src.definitions import FRAGMENT_LENGTH
from src.files import Files
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.preporcess_strategy.CNN_af_preprocess_strategy import CNNAFPreprocessStrategy
from src.neural_network.model.stategies.preporcess_strategy.preprocess_strategy_interface import IModelPreprocessStrategy
from src.neural_network.model.types import ModelTypes
from src.definitions import sr, frame_length, hop_length


class CNNModelPreprocessStrategy(IModelPreprocessStrategy):
  def __init__(self, model_type: ModelTypes, af_type: AFTypes):
    self.model_type = model_type
    self.af_type = af_type
    self.files = Files()
    self.loger = Logger('CNNModelPreprocessStrategy')
    self.af_strategy = CNNAFPreprocessStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
  def reshape(self, i):
    w, h = self.af_strategy.get_shape()
    return tf.reshape(i, (-1, w, h, 1))

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
  def get_audio_feature(self, i):
    return tf.py_function(self.af_strategy.get_audio_feature, [i], tf.float32)

  @tf.function
  def save_audio_feature(self, i, labels, label_names):
    tf.py_function(self.af_strategy.save_audio_feature, [i, labels, label_names], tf.float32)
    return i

  def preprocess(self, data_set_path: str, save_af: bool = False):
    self.af_strategy.set_strategy(self.af_type)
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
