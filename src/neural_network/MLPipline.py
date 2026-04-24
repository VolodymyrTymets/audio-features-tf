import os
import numpy as np
import tensorflow as tf

from src.audio_features.types import AFTypes
from src.files import Files
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.build_strategy.build_strategy_factory import BuildStrategyFactory
from src.neural_network.model.stategies.preporcess_strategy.preprocessor_strategy_factory import \
  PreprocessorStrategyFactory
from src.neural_network.model.types import ModelTypes
from src.definitions import DURATION, EPOCHS, sr as SR, frame_length, hop_length, SUB_EPOCHS

from src.neural_network.model.model_preprocessor import ModelPreprocessor
from src.neural_network.model.model_builder import ModelBuilder
from src.neural_network.model.model_evaluator import ModelEvaluator
from src.neural_network.model.mode_exporter import MoldeExporter
from src.audio_features.strategy.af_strategy_factory import AFStrategyFactory
from src.neural_network.model.model_record_evaluator import ModelRecordEvaluator
from src.neural_network.model.model_record_label import ModelRecordColorLabeler

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
    self.af_strategy = AFStrategyFactory(sr=SR, frame_length=frame_length, hop_length=hop_length).create_strategy(
      af_type)
    self.model_exporter = MoldeExporter(model_type=self.model_type, af_type=self.af_type)
    self.preprocessor_strategy = PreprocessorStrategyFactory(af_strategy=self.af_strategy).create_strategy(model_type)
    self.build_strategy = BuildStrategyFactory(af_strategy=self.af_strategy).create_strategy(model_type)

  def train(self, save_af=False, save_model=True):
    model_preprocessor = ModelPreprocessor(strategy=self.preprocessor_strategy)
    model_builder = ModelBuilder(strategy=self.build_strategy, target_path=self.model_exporter.export_path)
    mode_evaluator = ModelEvaluator()

    train_ds, val_ds, test_ds, label_names = model_preprocessor.preprocess(
      data_set_path=self.files.join(self.files.ASSETS_PATH, f'data_set_{DURATION}'), save_af=save_af
    )
    for example, example_spect_labels in train_ds.take(1):
      input_shape = example.shape[1:]
      model = model_builder.build(input_shape, output_shape=len(label_names), train_ds=train_ds)
      self.model_exporter.export_model_plot(model, target_path=os.path.join(self.model_exporter.export_path, 'model_plot.png'))

      for epochs in range(SUB_EPOCHS, EPOCHS + SUB_EPOCHS, SUB_EPOCHS):
        result = model_builder.train(train_ds, val_ds, epochs=epochs)

        if result is not None:
          self.model_exporter.export_metrics(result.history)
          self.model_exporter.export_training_plot()
          evaluation = mode_evaluator.evaluate(model=model, test_ds=test_ds)
          test_acc, test_loss = evaluation.values()
          test_record_acc = 0

          if save_model:
            epoch_path = os.path.join(self.model_exporter.export_path, 'epoch_' + str(epochs))
            target_path = os.path.join(epoch_path, 'model')
            self.model_exporter.export_model(model, label_names, input_shape, target_path=target_path)

            saved_model = self.model_exporter.load_model(target_path=target_path)
            record_evaluator = ModelRecordEvaluator(self.af_strategy, self.af_type, self.model_type, model=saved_model)
            test_record_acc = record_evaluator.evaluate_records()
            record_evaluator = ModelRecordColorLabeler(self.af_strategy, self.af_type, self.model_type, model=saved_model)
            record_evaluator.label_records(export_path=epoch_path)

          self.model_exporter.export_evaluation(epochs, record_acc=test_record_acc, acc=test_acc, loss=test_loss)

          if epochs == EPOCHS:
            self.model_exporter.export_model(model, label_names, input_shape,
                                             target_path=os.path.join(self.model_exporter.export_path, 'model'))
            self.model_exporter.export_evaluation_report(self.model_exporter.get_max_evaluation('acc'), self.af_type,
                                                         self.model_type, 'acc')
            self.model_exporter.export_evaluation_report(
            self.model_exporter.get_max_evaluation('loss', aggregation='min'), self.af_type, self.model_type, 'loss')
            self.model_exporter.export_evaluation_report(self.model_exporter.get_max_evaluation('record_acc'),
                                                         self.af_type, self.model_type, 'record_acc')
          self.loger.log(f'Test record accuracy: {test_record_acc}%', 'blue')
          self.loger.log('Model trained and evaluated', 'green')

  def label_records(self):
    model = self.model_exporter.load_model(target_path=os.path.join(self.model_exporter.export_path, 'model'))
    record_evaluator = ModelRecordColorLabeler(self.af_strategy, self.af_type, self.model_type, model=model)
    record_evaluator.label_records(self.model_exporter.export_path)

  def evaluate_records(self):
    model = self.model_exporter.load_model(target_path=os.path.join(self.model_exporter.export_path, 'model'))
    record_evaluator = ModelRecordEvaluator(self.af_strategy, self.af_type, self.model_type, model=model)

    test_record_acc = record_evaluator.evaluate_records()
    self.model_exporter.export_evaluation_report(self.model_exporter.get_max_evaluation('record_acc'), self.af_type,
                                                 self.model_type, 'record_acc')

    _, fragment_time, record_time = record_evaluator.time_records(shift=30)
    self.model_exporter.export_evaluation_report(fragment_time, self.af_type, self.model_type, 'fragment_time')
    self.model_exporter.export_evaluation_report(record_time, self.af_type, self.model_type, 'record_time')

    self.loger.log(f'Test record accuracy: {test_record_acc}%', 'blue')
    return test_record_acc
