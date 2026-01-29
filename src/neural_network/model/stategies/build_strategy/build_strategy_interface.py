from abc import ABC, abstractmethod

import numpy as np

class IModelBuildStrategy(ABC):
  @abstractmethod
  def build(self, input_shape: np.ndarray, output_shape: int, train_ds):
    pass
