from enum import Enum


class AFTypes(Enum):
  wave = 'wave'
  stft = 'stft'
  fft = 'fft'
  mel = 'mel'
  mfcc = 'mfcc'