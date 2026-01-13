
import numpy as np
from src.audio_features.audio_features import FrequencyDomainFeatures
from src.audio_features.types import AFTypes
from src.neural_network.strategy.strategies.wave_strategy import WaveStrategy
from src.neural_network.strategy.strategies.ae_strategy import AEStrategy
from src.neural_network.strategy.strategies.rms_strategy import RMStrategy
from src.neural_network.strategy.strategies.zcr_strategy import ZCRtrategy
from src.neural_network.strategy.strategies.fft_strategy import FFTStrategy
from src.neural_network.strategy.strategies.stft_strategy import STFTStrategy
from src.neural_network.strategy.strategies.ber_strategy import BERtrategy
from src.neural_network.strategy.strategies.sc_strategy import SCStrategy
from src.neural_network.strategy.strategies.bw_strategy import BWtrategy
from src.neural_network.strategy.strategies.mel_strategy import MelStrategy
from src.neural_network.strategy.strategies.mfcc_strategy import MFCCStrategy


class AFStrategy:
  def __init__(self, strategy_type: AFTypes, sr: int, frame_length: int, hop_length: int):
    self.features = FrequencyDomainFeatures()
    self.wave_strategy = WaveStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.ae_strategy = AEStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.rms_strategy = RMStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.zcr_strategy = ZCRtrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.fft_strategy = FFTStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.stft_strategy = STFTStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.ber_strategy = BERtrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.sc_strategy = SCStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.bw_strategy = BWtrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.mel_strategy = MelStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.mfcc_strategy = MFCCStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)

    if strategy_type.value == AFTypes.wave.value:
      self.strategy = self.wave_strategy
    elif strategy_type.value == AFTypes.ae.value:
      self.strategy = self.ae_strategy
    elif strategy_type.value == AFTypes.rms.value:
      self.strategy = self.rms_strategy
    elif strategy_type.value == AFTypes.zcr.value:
      self.strategy = self.zcr_strategy
    elif strategy_type.value == AFTypes.ae.value:
      self.strategy = self.ae_strategy
    elif strategy_type.value == AFTypes.fft.value:
      self.strategy = self.fft_strategy
    elif strategy_type.value == AFTypes.ber.value:
      self.strategy = self.ber_strategy
    elif strategy_type.value == AFTypes.sc.value:
      self.strategy = self.sc_strategy
    elif strategy_type.value == AFTypes.bw.value:
      self.strategy = self.bw_strategy
    elif strategy_type.value == AFTypes.stft.value:
      self.strategy = self.stft_strategy
    elif strategy_type.value == AFTypes.mel.value:
      self.strategy = self.mel_strategy
    elif strategy_type.value == AFTypes.mfcc.value:
      self.strategy = self.mfcc_strategy
    else:
      raise ValueError(f"Unknown strategy type: {strategy_type.value}")

  def get_audio_feature(self, signal: np.array):
    if self.strategy is None:
      raise ValueError("Strategy not set")
    return self.strategy.get_audio_feature(signal)
