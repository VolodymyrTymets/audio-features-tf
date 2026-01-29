from src.audio_features.types import AFTypes
from src.files import Files
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.build_strategy.CNN_strategy import CNNModelBuildStrategy
from src.neural_network.model.types import ModelTypes


class ModelBuilder:
  def __init__(self, model_type: ModelTypes, af_type: AFTypes):
    self.model_type = model_type
    self.af_type = af_type
    self.model = None
    self.build_strategy = None
    self.files = Files()
    self.loger = Logger('ModelBuilder')

    if model_type.value == ModelTypes.CNN.value:
      self.build_strategy = CNNModelBuildStrategy()

  def build(self, input_shape, output_shape, train_ds):
    self.loger.log(f'Building model with input shape: {input_shape}...')
    self.model = self.build_strategy.build(input_shape, output_shape, train_ds)
    self.loger.log(f'Model built: {self.model.name}', 'green')
    self.model.summary()
    return self.model

  def train(self, train_ds, val_ds, epochs):
    if self.model is None:
      raise ValueError("Model not set")

    self.loger.log(f'Training model: {self.model.name}')
    history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    self.loger.log(f'Model trained: {self.model.name}', 'green')
    return history





