from src.neural_network.model.stategies.preporcess_strategy.preprocess_strategy_interface import \
  IModelPreprocessStrategy


class ModelPreprocessor:
  def __init__(self, strategy: IModelPreprocessStrategy):
    self.strategy = strategy

  def preprocess(self, data_set_path: str, save_af: bool = False):
    return self.strategy.preprocess(data_set_path, save_af)





