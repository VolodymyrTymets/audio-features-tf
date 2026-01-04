from abc import ABC, abstractmethod

import numpy as np

class IAFStrategy(ABC):
  @abstractmethod
  def get_audio_feature(self, wave: np.ndarray):
    pass
