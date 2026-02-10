import numpy as np
import tensorflow as tf

from src.audio_features.types import AFTypes
from src.files import Files
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.build_strategy.build_strategy_factory import BuildStrategyFactory
from src.neural_network.model.stategies.preporcess_strategy.preprocessor_strategy_factory import \
  PreprocessorStrategyFactory
from src.neural_network.model.types import ModelTypes
from src.definitions import DURATION, EPOCHS, sr as SR, frame_length, hop_length

from src.neural_network.model.model_preprocessor import ModelPreprocessor
from src.neural_network.model.model_builder import ModelBuilder
from src.neural_network.model.model_evaluator import ModelEvaluator
from src.neural_network.model.mode_exporter import MoldeExporter
from src.audio_features.strategy.af_strategy_factory import AFStrategyFactory
from src.neural_network.model.model_record_evaluator import ModelRecordEvaluator, ModelRecordColorLabeler


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
    self._model_dir = self.files.join(self.files.ASSETS_PATH, 'models',
                                      'm_{}_{}_{}'.format(DURATION, self.af_type.value, self.model_type.value))
    self.af_strategy = AFStrategyFactory(sr=SR, frame_length=frame_length, hop_length=hop_length).create_strategy(
      af_type)
    self.preprocessor_strategy = PreprocessorStrategyFactory(af_strategy=self.af_strategy).create_strategy(model_type)
    self.build_strategy = BuildStrategyFactory(af_strategy=self.af_strategy).create_strategy(model_type)


  def train(self, save_af=False, save_model=True):
    model_preprocessor = ModelPreprocessor(strategy=self.preprocessor_strategy)
    model_builder = ModelBuilder(strategy=self.build_strategy, target_path=self._model_dir)
    mode_evaluator = ModelEvaluator()
    model_exporter = MoldeExporter(model_type=self.model_type, af_type=self.af_type)

    train_ds, val_ds, test_ds, label_names = model_preprocessor.preprocess(
      data_set_path=self.files.join(self.files.ASSETS_PATH, f'data_set_{DURATION}'), save_af=save_af
    )
    for example, example_spect_labels in train_ds.take(1):
      input_shape = example.shape[1:]

      model = model_builder.build(input_shape, output_shape=len(label_names), train_ds=train_ds)
      result = model_builder.train(train_ds, val_ds, epochs=EPOCHS)

      if result is not None:
        model_exporter.export_metrics(result.history)
        model_exporter.export_training_plot()

      if save_model:
        model_exporter.export_model(model, label_names, input_shape, target_path=self._model_dir)

      evaluation = mode_evaluator.evaluate(model=model, test_ds=test_ds)
      test_acc, test_loss = evaluation.values()
      record_evaluator = ModelRecordEvaluator(self.af_strategy, self.af_type, self.model_type)
      test_record_acc = record_evaluator.evaluate_record('test.wav')


      model_exporter.export_evaluation(test_acc, self.af_type, self.model_type, 'acc')
      model_exporter.export_evaluation(test_loss, self.af_type, self.model_type, 'loss')
      model_exporter.export_evaluation(test_record_acc, self.af_type, self.model_type, 'record_acc')
      self.loger.log(f'Test record accuracy: {test_record_acc}%', 'blue')
      self.loger.log('Model trained and evaluated', 'green')

  def label_records(self):
    record_evaluator = ModelRecordColorLabeler(self.af_strategy, self.af_type, self.model_type, model=None)
    record_evaluator.label_records()

  def evaluate_record(self, file_name: str):
    record_evaluator = ModelRecordEvaluator(self.af_strategy, self.af_type, self.model_type, model=None)
    return record_evaluator.evaluate_record(file_name)




