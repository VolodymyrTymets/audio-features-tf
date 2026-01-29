
import numpy as np
import tensorflow as tf

from src.audio_features.types import AFTypes
from src.files import Files
from src.neural_network.model.types import ModelTypes
from src.definitions import DURATION, EPOCHS

from src.neural_network.model.model_preprocessor import ModelPreprocessor
from src.neural_network.model.model_builder import ModelBuilder
from src.neural_network.model.model_evaluator import ModelEvaluator
from src.neural_network.model.mode_exporter import MoldeExporter


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
files = Files()


def train(af_type: AFTypes, model_type: ModelTypes.CNN, save_af=False):
  model_preprocessor = ModelPreprocessor(model_type=model_type, af_type=af_type)
  model_builder = ModelBuilder(model_type=model_type, af_type=af_type)
  mode_evaluator = ModelEvaluator()
  mode_exporter = MoldeExporter(model_type=model_type, af_type=af_type)


  train_ds, val_ds, test_ds, label_names = model_preprocessor.preprocess(
    data_set_path=files.join(files.ASSETS_PATH, f'data_set_{DURATION}'), save_af=False
  )

  for example, example_spect_labels in train_ds.take(1):
    input_shape = example.shape[1:]

    model = model_builder.build(input_shape, output_shape=len(label_names), train_ds=train_ds)
    history = model_builder.train(train_ds, val_ds, epochs=EPOCHS)

    mode_evaluator.evaluate(model=model, test_ds=test_ds)

    mode_exporter.export_model(model, label_names, input_shape)
    mode_exporter.export_history(history)
    mode_exporter.export_metrics(history)




