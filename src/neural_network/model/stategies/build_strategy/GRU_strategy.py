
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from src.neural_network.model.stategies.build_strategy.build_strategy_interface import IModelBuildStrategy


class GRUModelBuildStrategy(IModelBuildStrategy):
  def build(self, input_shape: np.ndarray, output_shape: int, train_ds):
    model = models.Sequential([
      layers.Input(shape=input_shape),
      layers.GRU(32, return_sequences=True),
      layers.Dropout(0.25),
      layers.GRU(32),
      layers.Dropout(0.25),
      # # Flatten the result to feed into DNN
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(output_shape, activation='softmax'),
    ])
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
    )
    return model
