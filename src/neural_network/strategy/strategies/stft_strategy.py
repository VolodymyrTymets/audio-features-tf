import numpy as np

from src.audio_features.audio_features import FrequencyDomainFeatures
from src.neural_network.strategy.strategies.strategy_interface import IAFStrategy


class STFTStrategy(IAFStrategy):
  def __init__(self, frame_length: int, hop_length: int):
    self.features = FrequencyDomainFeatures()
    self.frame_length = frame_length
    self.hop_length = hop_length

  def get_audio_feature(self, wave: np.ndarray):
    return self.features.stft(signal=wave, frame_length=self.frame_length, hop_length=self.hop_length)
