from abc import ABC, abstractmethod

class ITrainStrategy(ABC):
  @abstractmethod
  def get_audio_feature(self, wave):
    pass

  @abstractmethod
  def reshape(self, wave):
    pass