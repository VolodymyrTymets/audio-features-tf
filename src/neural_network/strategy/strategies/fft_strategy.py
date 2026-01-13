import numpy as np
import matplotlib.pyplot as plt
from src.neural_network.strategy.strategies.base_strategy import BaseStrategy
from src.audio_features.types import AFTypes


class FFTStrategy(BaseStrategy):
  def __init__(self, sr: int, frame_length: int, hop_length: int):
    super(self.__class__, self).__init__(sr, frame_length, hop_length)
    self.af_type = AFTypes.fft
    fig, ax = plt.subplots()
    self.fig = fig
    self.ax = ax

  def get_audio_feature(self, wave: np.ndarray):
    matrix, freqs, bins, im = self.ax.specgram(wave, Fs=self.sr, NFFT=self.frame_length, cmap='plasma')
    return matrix
