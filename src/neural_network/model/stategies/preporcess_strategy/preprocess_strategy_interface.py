from abc import ABC, abstractmethod

class IModelPreprocessStrategy(ABC):
  @abstractmethod
  def preprocess(self, data_set_path: str, save_af: bool = False):
    pass
