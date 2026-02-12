
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from src.neural_network.model.stategies.build_strategy.build_strategy_interface import IModelBuildStrategy


class LSTMModelBuildStrategy(IModelBuildStrategy):
  def _get_dimension(self, input_shape):
    return len(input_shape[:-1])

  def _get_1d_layer(self):
    return [
      layers.Conv1D(32, kernel_size=8,
               strides=1,
               activation='relu',
               padding='same'),
      layers.LSTM(32, return_sequences=True),
      layers.LSTM(32),
    ]

  def _get_2d_layer(self):
    return [
      layers.LSTM(32, return_sequences=True),
      layers.LSTM(32),
    ]

  def build(self, input_shape: np.ndarray, output_shape: int, train_ds):
    norm_layer = layers.Normalization()
    norm_layer.adapt(data=train_ds.map(
      map_func=lambda spec, label: spec))

    model_dimension = self._get_dimension(input_shape)

    dimension_layers = []
    if model_dimension == 1:
      dimension_layers = self._get_1d_layer()
    elif model_dimension == 2:
      dimension_layers = self._get_2d_layer()

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(norm_layer, 'normalization')

    for layer in dimension_layers:
      model.add(layer)

    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='tanh'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_shape, activation='softmax'))

    model.get_layer('normalization').adapt(train_ds.map(lambda x, label: x))

    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
    )
    return model
