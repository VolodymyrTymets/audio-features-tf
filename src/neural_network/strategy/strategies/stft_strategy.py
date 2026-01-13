import numpy as np
from src.neural_network.strategy.strategies.base_strategy import BaseStrategy
from src.audio_features.types import AFTypes


class STFTStrategy(BaseStrategy):
  def __init__(self, sr: int, frame_length: int, hop_length: int):
    super(self.__class__, self).__init__(sr, frame_length, hop_length)
    self.af_type = AFTypes.stft

  def get_audio_feature(self, wave: np.ndarray):
    return self.features.stft(signal=wave, frame_length=self.frame_length, hop_length=self.hop_length)
