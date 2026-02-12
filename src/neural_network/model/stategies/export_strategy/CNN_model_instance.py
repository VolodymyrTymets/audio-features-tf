import tensorflow as tf


class CNNExportModelInstance(tf.Module):
  def __init__(self, model, input_shape, label_names):
    self.model = model
    self.label_names = label_names
    self.input_shape = input_shape

    self.__call__.get_concrete_function(
      x=tf.TensorSpec(shape=self._get_shape(input_shape), dtype=tf.float32))

  def _get_dimension(self, input_shape):
    return len(input_shape[:-1])

  def _get_shape(self, x):
    if self._get_dimension(x) == 1:
      return [x[0]]
    if self._get_dimension(x) == 2:
      return [x[0], x[1]]
    return x

  @tf.function
  def __call__(self, x):
    x = tf.reshape(x, [-1] + self._get_shape(self.input_shape) + [1])
    result = self.model(x, training=False)

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(self.label_names, class_ids)
    return {'predictions': result,
            'class_ids': class_ids,
            'class_names': class_names,
            'label_names': self.label_names}
