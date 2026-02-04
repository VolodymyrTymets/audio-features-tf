
import numpy as np
import tensorflow as tf

from src.audio_features.types import AFTypes
from src.files import Files
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.build_strategy.build_strategy_factory import BuildStrategyFactory
from src.neural_network.model.stategies.preporcess_strategy.preprocessor_strategy_factory import PreprocessorStrategyFactory
from src.neural_network.model.types import ModelTypes
from src.definitions import DURATION, EPOCHS, sr as SR, frame_length, hop_length

from src.neural_network.model.model_preprocessor import ModelPreprocessor
from src.neural_network.model.model_builder import ModelBuilder
from src.neural_network.model.model_evaluator import ModelEvaluator
from src.neural_network.model.mode_exporter import MoldeExporter
from src.audio_features.strategy.af_strategy_factory import AFStrategyFactory


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

class MLPipeline:
  def __init__(self, af_type: AFTypes, model_type: ModelTypes.CNN):
    self.af_type = af_type
    self.model_type = model_type
    self.files = Files()
    self.loger = Logger('MLPipeline')
    self.af_strategy = AFStrategyFactory(sr=SR, frame_length=frame_length, hop_length=hop_length).create_strategy(af_type)
    self.preprocessor_strategy = PreprocessorStrategyFactory(af_strategy=self.af_strategy).create_strategy(model_type)
    self.build_strategy = BuildStrategyFactory(af_strategy=self.af_strategy).create_strategy(model_type)


  def train(self, save_af=False):
    model_preprocessor = ModelPreprocessor(strategy=self.preprocessor_strategy)
    model_builder = ModelBuilder(strategy=self.build_strategy)
    mode_evaluator = ModelEvaluator()
    model_exporter = MoldeExporter(model_type=self.model_type, af_type=self.af_type)

    train_ds, val_ds, test_ds, label_names = model_preprocessor.preprocess(
      data_set_path=self.files.join(self.files.ASSETS_PATH, f'data_set_{DURATION}'), save_af=save_af
    )

    for example, example_spect_labels in train_ds.take(1):
      input_shape = example.shape[1:]
      print(f'Input shape: {input_shape}')

      model = model_builder.build(input_shape, output_shape=len(label_names), train_ds=train_ds)
      history = model_builder.train(train_ds, val_ds, epochs=EPOCHS)

      evaluation = mode_evaluator.evaluate(model=model, test_ds=test_ds)
      test_acc, test_loss = evaluation.values()

      model_exporter.export_model(model, label_names, input_shape)
      model_exporter.export_history(history)
      model_exporter.export_metrics(history)
      model_exporter.export_evaluation(test_acc, self.af_type, self.model_type, 'acc')
      model_exporter.export_evaluation(test_loss, self.af_type, self.model_type, 'loss')




