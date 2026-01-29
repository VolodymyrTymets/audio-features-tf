
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.audio_features.types import AFTypes
from src.definitions import FRAGMENT_LENGTH
from src.files import Files
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.preporcess_strategy.preprocess_strategy_interface import IModelPreprocessStrategy
from src.neural_network.model.types import ModelTypes


class CNNModelPreprocessStrategy(IModelPreprocessStrategy):
  def __init__(self, model_type: ModelTypes, af_type: AFTypes, af_train_strategy):
    self.model_type = model_type
    self.af_type = af_type
    self.files = Files()
    self.loger = Logger('CNNModelPreprocessStrategy')
    self.train_strategy = af_train_strategy

  def preprocess(self, data_set_path: str, save_af: bool = False):
    self.train_strategy.set_strategy(self.af_type)
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
    train_ds = train_ds.map(self.train_strategy.squeeze_map, tf.data.AUTOTUNE)
    val_ds = val_ds.map(self.train_strategy.squeeze_map, tf.data.AUTOTUNE)
    test_ds = test_ds.map(self.train_strategy.squeeze_map, tf.data.AUTOTUNE)
    # val_ds = val_ds.shard(num_shards=2, index=1)

    train_ds = train_ds.map(self.train_strategy.get_audio_feature_map, tf.data.AUTOTUNE)
    val_ds = val_ds.map(self.train_strategy.get_audio_feature_map, tf.data.AUTOTUNE)
    test_ds = test_ds.map(self.train_strategy.get_audio_feature_map, tf.data.AUTOTUNE)

    if save_af:
      train_ds = train_ds.map(self.train_strategy.save_audio_feature_map, tf.data.AUTOTUNE)

    train_ds = train_ds.map(self.train_strategy.reshape_map, tf.data.AUTOTUNE)
    val_ds = val_ds.map(self.train_strategy.reshape_map, tf.data.AUTOTUNE)
    test_ds = test_ds.map(self.train_strategy.reshape_map, tf.data.AUTOTUNE)

    # Training model
    train_ds = train_ds.cache().shuffle(
      10000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, label_names
