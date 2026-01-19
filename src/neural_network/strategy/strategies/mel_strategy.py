import numpy as np

from src.definitions import n_mels
from src.neural_network.strategy.strategies.base_strategy import BaseStrategy
from src.audio_features.types import AFTypes


class MelStrategy(BaseStrategy):
  def __init__(self, sr: int, frame_length: int, hop_length: int, sub_folder: str, n_mels: int = n_mels):
    super(self.__class__, self).__init__(sr, frame_length, hop_length, sub_folder)
    self.af_type = AFTypes.mel
    self.n_mels = n_mels

  def get_audio_feature(self, wave: np.ndarray):
    return self.features.melspectogram(signal=wave, sr=self.sr, frame_length=self.frame_length,
                                       hop_length=self.hop_length, n_mels=self.n_mels)
