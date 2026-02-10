
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from src.neural_network.model.stategies.build_strategy.build_strategy_interface import IModelBuildStrategy


class GRUModelBuildStrategy(IModelBuildStrategy):
  def build(self, input_shape: np.ndarray, output_shape: int, train_ds):
    norm_layer = layers.Normalization()
    norm_layer.adapt(data=train_ds.map(
      map_func=lambda spec, label: spec))

    model = models.Sequential([
      layers.Input(shape=input_shape),
      norm_layer,
      layers.GRU(32, return_sequences=True),
      layers.GRU(64),
      layers.Dropout(0.2),
      layers.Flatten(),
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
