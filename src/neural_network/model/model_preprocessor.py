from src.audio_features.types import AFTypes
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.preporcess_strategy.CNN_strategy import CNNModelPreprocessStrategy
from src.neural_network.model.types import ModelTypes
from src.neural_network.strategy.train_strategy import TrainStrategy
from src.definitions import sr, frame_length, hop_length


class ModelPreprocessor:
  def __init__(self, model_type: ModelTypes, af_type: AFTypes):
    self.model_type = model_type
    self.af_type = af_type
    self.train_strategy = TrainStrategy(label_names=[], sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.loger = Logger('ModelBuilder')
    if model_type.value == ModelTypes.CNN.value:
      self.strategy = CNNModelPreprocessStrategy(model_type=model_type, af_type=af_type,
                                                 af_train_strategy=self.train_strategy)

  def preprocess(self, data_set_path: str, save_af: bool = False):
    return self.strategy.preprocess(data_set_path, save_af=save_af)
