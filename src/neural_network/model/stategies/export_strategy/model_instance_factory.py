import tensorflow as tf

from src.neural_network.model.stategies.export_strategy.CNN_model_instance import CNNExportModelInstance
from src.neural_network.model.types import ModelTypes


class ModelInstanceFactory:
  def __init__(self, model_type: ModelTypes):
    self.model_type = model_type

  def create_model_instance(self, model, label_names, input_shape):
    if self.model_type.value == ModelTypes.CNN.value:
      return CNNExportModelInstance(model=model, label_names=label_names, input_shape=input_shape)
    else:
      raise ValueError(f'Model type {self.model_type.value} not supported')
