import random
import numpy as np
import pyloudnorm as pyln


class DataTransformer:
  def __init__(self, sr: int, frame_length: int, hop_length: int):
    self.sr = sr
    self.frame_length = frame_length
    self.hop_length = hop_length

  def calc_fragment_size(self, duration: float):
    return int(self.sr / (1 / duration))

  def split(self, signal: np.array,  duration: float):
    fragment_size = self.calc_fragment_size(duration)
    fragment_count = len(signal) // fragment_size
    fragments = []
    for i in range(fragment_count):
      fragment = signal[i * fragment_size:(i + 1) * fragment_size]
      if len(fragment) < fragment_size:
        fragment += np.zeros(fragment_size - len(fragment))
      fragments.append(fragment)
    return np.array(fragments)

  def normalize(self, signal: np.array):
      # meter = pyln.Meter(self.sr) # create BS.1770 meter
      # loudness = meter.integrated_loudness(np.concatenate([signal, np.zeros(self.sr * 2)]))  # measure loudness range
      new_loudness = [
        (-60 * (random.randint(1, 50) / 100)),
        (-1 * (random.randint(1, 50) / 100))
      ]
      return [
        # pyln.normalize.loudness(signal, loudness, new_loudness[0]),
        # pyln.normalize.loudness(signal, loudness, new_loudness[1])
        pyln.normalize.peak(signal, new_loudness[0]),
        pyln.normalize.peak(signal, new_loudness[1]),
      ]


