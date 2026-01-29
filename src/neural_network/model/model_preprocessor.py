from src.audio_features.types import AFTypes
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.preporcess_strategy.CNN_strategy import CNNModelPreprocessStrategy
from src.neural_network.model.types import ModelTypes


class ModelPreprocessor:
  def __init__(self, model_type: ModelTypes, af_type: AFTypes):
    self.model_type = model_type
    self.af_type = af_type
    self.loger = Logger('ModelBuilder')
    if model_type.value == ModelTypes.CNN.value:
      self.strategy = CNNModelPreprocessStrategy(model_type=model_type, af_type=af_type)

  def preprocess(self, data_set_path: str, save_af: bool = False):
    return self.strategy.preprocess(data_set_path, save_af=save_af)
