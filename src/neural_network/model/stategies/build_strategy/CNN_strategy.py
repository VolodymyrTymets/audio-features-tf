import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from src.neural_network.model.stategies.build_strategy.build_strategy_interface import IModelBuildStrategy


class CNNModelBuildStrategy(IModelBuildStrategy):
  def _get_dimension(self, input_shape):
    return len(input_shape[:-1])

  def _get_1d_layer(self):
    return [
      # Normalize.
      layers.Normalization(name='normalization'),
      layers.Conv1D(32, (3,), activation='relu'),
      layers.Conv1D(64, (3,), activation='relu'),
      layers.MaxPooling1D()
    ]

  def _get_2d_layer(self):
    return [
      # Downsample the input.
      layers.Resizing(32, 32),
      # Normalize.
      layers.Normalization(name='normalization'),
      layers.Conv2D(32, 3, activation='relu'),
      layers.Conv2D(64, 3, activation='relu'),
      layers.MaxPooling2D()
    ]

  def build(self, input_shape: np.ndarray, output_shape: int, train_ds):
    model_dimension = self._get_dimension(input_shape)

    dimension_layers = []
    if model_dimension == 1:
      dimension_layers = self._get_1d_layer()
    elif model_dimension == 2:
      dimension_layers = self._get_2d_layer()

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    for layer in dimension_layers:
      model.add(layer)

    model.add(layers.Dropout(0.2))
    # Flatten the result to feed into DNN
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_shape, activation='softmax'))

    model.get_layer('normalization').adapt(train_ds.map(lambda x, label: x))

    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
    )
    return model
