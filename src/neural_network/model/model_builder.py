import os
import pathlib
import numpy as np
import tensorflow as tf

from src.audio_features.types import AFTypes
from src.files import Files
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.CNN_strategy import CNNModelBuildStrategy
from src.neural_network.model.types import ModelTypes
from src.neural_network.strategy.train_strategy import TrainStrategy
from src.definitions import FRAGMENT_LENGTH, sr, frame_length, hop_length, DURATION
from src.neural_network.model.model_export import ExportModel


class ModelBuilder:
  def __init__(self, model_type: ModelTypes, af_type: AFTypes):
    self.model_type = model_type
    self.af_type = af_type
    self.model = None
    self.train_ds = None
    self.build_strategy = None
    self.train_strategy = None
    self.files = Files()
    self.train_strategy = TrainStrategy(label_names=[], sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.loger = Logger('ModelBuilder')

    if model_type.value == ModelTypes.CNN.value:
      self.build_strategy = CNNModelBuildStrategy()

  def make_data_set(self, data_set_name: str, save_af: bool = False):
    self.train_strategy.set_strategy(self.af_type)
    # Form data storage
    data_dir = self.files.join(self.files.ASSETS_PATH, data_set_name)
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
      directory=self.files.join(data_dir, 'train'),
      batch_size=32,
      validation_split=0.2,
      seed=0,
      output_sequence_length=FRAGMENT_LENGTH,
      subset='both')
    test_ds = tf.keras.utils.audio_dataset_from_directory(
      directory=self.files.join(data_dir, 'test'),
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
    self.train_ds = train_ds

    return train_ds, val_ds, test_ds, label_names


  def build(self, input_shape, output_shape):
    self.loger.log(f'Building model with input shape: {input_shape}...')
    self.model = self.build_strategy.build(input_shape, output_shape, self.train_ds)
    self.loger.log(f'Model built: {self.model.name}', 'green')
    self.model.summary()
    return self.model

  def train(self, train_ds, val_ds, epochs):
    if self.model is None:
      raise ValueError("Model not set")

    self.loger.log(f'Training model: {self.model.name}')
    history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    self.loger.log(f'Model trained: {self.model.name}', 'green')
    return history

  def evaluate(self, test_ds):
    if self.model is None:
      raise ValueError("Model not set")
    self.loger.log('Evaluation of model:', 'green')
    self.model.evaluate(test_ds, return_dict=True)

  def save(self, label_names, input_shape):
    self.loger.log('Saving model...', 'blue')
    model_dir = self.files.join(self.files.ASSETS_PATH, 'models', 'm_{}_{}_{}'.format(DURATION, self.af_type.value, self.model_type.value))
    export = ExportModel(model=self.model, label_names=label_names, input_shape=input_shape)
    tf.saved_model.save(export, model_dir)
    self.loger.log('Model is saved to: {}'.format(model_dir), 'green')

