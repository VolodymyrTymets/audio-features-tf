
import numpy as np
from src.audio_features.audio_features import FrequencyDomainFeatures
from src.audio_features.types import AFTypes
from src.neural_network.strategy.strategies.fft_strategy import FFTStrategy
from src.neural_network.strategy.strategies.stft_strategy import STFTStrategy
from src.neural_network.strategy.strategies.mel_strategy import MelStrategy


class AFStrategy:
  def __init__(self, strategy_type: AFTypes, sr: int, frame_length: int, hop_length: int):
    self.features = FrequencyDomainFeatures()
    self.stft_strategy = STFTStrategy(frame_length, hop_length)
    self.fft_strategy = FFTStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.mel_strategy = MelStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    if strategy_type.value == AFTypes.stft.value:
      self.strategy = self.stft_strategy
    elif strategy_type.value == AFTypes.fft.value:
      self.strategy = self.fft_strategy
    elif strategy_type.value == AFTypes.mel.value:
      self.strategy = self.mel_strategy
    else:
      raise ValueError(f"Unknown strategy type: {strategy_type.value}")

  def get_audio_feature(self, signal: np.array):
    if self.strategy is None:
      raise ValueError("Strategy not set")
    return self.strategy.get_audio_feature(signal)
