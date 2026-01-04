import tensorflow as tf
from src.neural_network.strategy.af_strategy import AF_Stratedgy
from src.definitions import sr, FRAGMENT_LENGTH


class ExportModel(tf.Module):
  def __init__(self, model, strategy: AF_Stratedgy, label_names, fragment_length=FRAGMENT_LENGTH, ):
    self.model = model
    self.label_names = label_names
    self.fragment_length = fragment_length
    self._strategy = strategy

    self.__call__.get_concrete_function(
      x=tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32))

  @tf.function
  def __call__(self, x):
    x = self._strategy.get_audio_feature(x)
    x = self._strategy.reshape(x)
    result = self.model(x, training=False)

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(self.label_names, class_ids)
    return {'predictions': result,
            'class_ids': class_ids,
            'class_names': class_names,
            'label_names': self.label_names}
