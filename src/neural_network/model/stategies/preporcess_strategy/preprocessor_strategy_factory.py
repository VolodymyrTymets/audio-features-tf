from src.neural_network.model.stategies.preporcess_strategy.CNN_strategy import CNNModelPreprocessStrategy
from src.neural_network.model.types import ModelTypes
from src.audio_features.strategy.strategies.strategy_interface import IAFStrategy


class PreprocessorStrategyFactory:
  def __init__(self, af_strategy: IAFStrategy):
    self.af_strategy = af_strategy

  def create_strategy(self, model_type: ModelTypes):
    if model_type.value == ModelTypes.CNN.value:
      return CNNModelPreprocessStrategy(af_strategy=self.af_strategy)
    else:
      raise ValueError(f'Model type {model_type.value} not supported')
