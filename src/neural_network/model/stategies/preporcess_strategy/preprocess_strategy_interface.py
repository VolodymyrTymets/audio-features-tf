from abc import ABC, abstractmethod

from src.audio_features.strategy.strategies.strategy_interface import IAFStrategy


class IModelPreprocessStrategy(ABC):
  @abstractmethod
  def __init__(self, af_strategy: IAFStrategy):
    pass

  @abstractmethod
  def preprocess(self, data_set_path: str, save_af: bool = False):
    pass
