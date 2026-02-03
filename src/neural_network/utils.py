import numpy as np
import tensorflow as tf



def label_by_model(model, input_x: np.ndarray):
  result = model(tf.convert_to_tensor(input_x, dtype=tf.float32))
  label = result['label_names'].numpy()[result['class_ids'].numpy()[0]]
  prediction = result['predictions'].numpy()[0][result['class_ids'].numpy()[0]]
  return label, prediction