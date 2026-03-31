from enum import Enum


class ModelTypes(Enum):
  ANN = 'ANN'
  CNN = 'CNN'
  LSTM = 'LSTM'
  GRU = 'GRU'
  CUSTOM = 'CUSTOM'