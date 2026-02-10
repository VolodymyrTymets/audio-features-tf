
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from src.neural_network.model.stategies.build_strategy.build_strategy_interface import IModelBuildStrategy


class CNNModelBuildStrategy(IModelBuildStrategy):
  def build(self, input_shape: np.ndarray, output_shape: int, train_ds):
    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train_ds.map(
      map_func=lambda spec, label: spec))

    model = models.Sequential([
      layers.Input(shape=input_shape),
      # Downsample the input.
      layers.Resizing(32, 32),
      # Normalize.
      norm_layer,
      layers.Conv2D(32, 3, activation='relu'),
      layers.Conv2D(64, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      # # Flatten the result to feed into DNN
      layers.Flatten(),
      # layers.Dense(512, activation='relu'),
      # layers.Dropout(0.2),
      # layers.Dense(128, activation='relu'),
      # layers.Dropout(0.3),
      layers.Dense(64, activation='tanh'),
      layers.Dropout(0.5),
      layers.Dense(output_shape, activation='softmax'),
    ])
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
    )
    return model
