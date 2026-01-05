from enum import Enum


class AFTypes(Enum):
  stft = 'stft'
  fft = 'fft'
  melspectogram = 'melspectogram'