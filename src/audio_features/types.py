from enum import Enum


class AFTypes(Enum):
  stft = 'stft'
  fft = 'fft'
  mel = 'mel'
  mfcc = 'mfcc'