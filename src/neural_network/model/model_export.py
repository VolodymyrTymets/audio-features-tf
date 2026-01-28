import tensorflow as tf
from src.neural_network.strategy.train_strategy_interface import ITrainStrategy
from src.definitions import sr, FRAGMENT_LENGTH


class ExportModel(tf.Module):
  def __init__(self, model, input_shape, label_names,):
    self.model = model
    self.label_names = label_names
    self.input_shape = input_shape
    w, h = self.input_shape[0], self.input_shape[1]
    self.__call__.get_concrete_function(
      x=tf.TensorSpec(shape=(w, h), dtype=tf.float32))

  @tf.function
  def __call__(self, x):
    w, h = self.input_shape[0], self.input_shape[1]
    x = tf.reshape(x, (-1, w, h, 1))
    result = self.model(x, training=False)

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(self.label_names, class_ids)
    return {'predictions': result,
            'class_ids': class_ids,
            'class_names': class_names,
            'label_names': self.label_names}
