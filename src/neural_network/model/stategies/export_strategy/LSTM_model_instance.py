import tensorflow as tf


class LSTMExportModelInstance(tf.Module):
  def __init__(self, model, input_shape, label_names):
    self.model = model
    self.label_names = label_names
    self.input_shape = input_shape
    self.__call__.get_concrete_function(
      x=tf.TensorSpec(shape=input_shape, dtype=tf.float32))

  @tf.function
  def __call__(self, x):
    w, h = self.input_shape[0], self.input_shape[1]
    x = tf.reshape(x, (-1, w, h))
    result = self.model(x, training=False)

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(self.label_names, class_ids)
    return {'predictions': result,
            'class_ids': class_ids,
            'class_names': class_names,
            'label_names': self.label_names}
