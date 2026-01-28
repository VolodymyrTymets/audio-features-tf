
import numpy as np
import tensorflow as tf

from src.audio_features.types import AFTypes
from src.neural_network.model.types import ModelTypes
from src.definitions import DURATION, EPOCHS

from src.neural_network.model.model_builder import ModelBuilder
from src.neural_network.model.mode_metrics import MoldeMetrics


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def train(af_type: AFTypes, model_type: ModelTypes.CNN, save_af=False):
  model_builder = ModelBuilder(model_type=model_type, af_type=af_type)
  mode_metrics = MoldeMetrics(model_type=model_type, af_type=af_type)

  train_ds, val_ds, test_ds, label_names = model_builder.make_data_set(
    data_set_name=f'data_set_{DURATION}', save_af=save_af
  )

  for example, example_spect_labels in train_ds.take(1):
    input_shape = example.shape[1:]

    model_builder.build(input_shape, output_shape=len(label_names))
    history = model_builder.train(train_ds, val_ds, epochs=EPOCHS)
    model_builder.evaluate(test_ds)
    model_builder.save(label_names, input_shape)

    mode_metrics.save_history(history)
    mode_metrics.save_metrics(history)




