from abc import ABC, abstractmethod

class IAFPreprocessStrategy(ABC):
  @abstractmethod
  def get_audio_feature(self, wave):
    pass

  @abstractmethod
  def save_audio_feature(self, af, label):
    pass

  @abstractmethod
  def get_shape(self):
    pass