from src.logger.logger_service import Logger
from src.neural_network.model.stategies.build_strategy.build_strategy_interface import IModelBuildStrategy

class ModelBuilder:
  def __init__(self, strategy: IModelBuildStrategy):
    self.model = None
    self.strategy = strategy
    self.loger = Logger('ModelBuilder')

  def build(self, input_shape, output_shape, train_ds):
    self.loger.log(f'Building model with input shape: {input_shape}...')
    self.model = self.strategy.build(input_shape, output_shape, train_ds)
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





