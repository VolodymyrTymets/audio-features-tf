import os
import pathlib
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import models
from matplotlib import pyplot as plt

from src.audio_features.types import AFTypes
from src.neural_network.model import ExportModel
from src.neural_network.strategy.train_strategy import TrainStrategy
from src.definitions import DURATION, FRAGMENT_LENGTH, EPOCHS, ASSETS_PATH, sr, frame_length, hop_length
from src.files import Files

files = Files()

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def save_history(history, metrics, af_type):
  plt.figure(figsize=(16, 6))
  plt.subplot(1, 2, 1)
  plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
  plt.legend(['loss', 'val_loss'])
  plt.ylim([0, max(plt.ylim())])
  plt.xlabel('Epoch')
  plt.ylabel('Loss [CrossEntropy]')

  plt.subplot(1, 2, 2)
  plt.plot(history.epoch, 100 * np.array(metrics['accuracy']), 100 * np.array(metrics['val_accuracy']))
  plt.legend(['accuracy', 'val_accuracy'])
  plt.ylim([0, 100])
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy [%]')

  dir_name = files.join(files.ASSETS_PATH, '__af__', af_type.value)
  files.create_folder(dir_name)
  plt.savefig(files.join(dir_name, 'training_history.png'))

def save_metrics(metrics, af_type):
  loss = metrics['loss']
  val_loss = metrics['val_loss']
  accuracy = metrics['accuracy'] * 100
  val_accuracy = metrics['val_accuracy'] * 100
  data = [
    ['loss', 'val_loss', 'accuracy', 'val_accuracy'],
  ] + [[los, val_loss[i], accuracy[i], val_accuracy[i]] for i, los in enumerate(loss) ]
  print('Saving report...')
  dir_path = files.join(files.ASSETS_PATH, '__af__', af_type.value)
  files.create_folder(dir_path)

  with open(files.join(dir_path, f'{af_type.value}_metics.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
  print('Report saved!')



def train(af_type: AFTypes, save_af=False):
  print(f'Training model for {af_type.value} audio_feature...', tf.executing_eagerly())
  af_type_value = af_type.value

  # Form data storage
  data_dir = pathlib.Path(os.path.join(ASSETS_PATH, 'data_set_{}'.format(DURATION)))
  train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=os.path.join(data_dir, 'train'),
    batch_size=32,
    validation_split=0.2,
    seed=0,
    output_sequence_length=FRAGMENT_LENGTH,
    subset='both')
  test_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=os.path.join(data_dir, 'valid'),
    batch_size=32,
    seed=0,
    output_sequence_length=FRAGMENT_LENGTH)

  label_names = np.array(train_ds.class_names)

  strategy = TrainStrategy(label_names=label_names, sr=sr, frame_length=frame_length, hop_length=hop_length)
  strategy.set_strategy(af_type)

  # Prepare data - wave to audio feature
  train_ds = train_ds.map(strategy.squeeze_map, tf.data.AUTOTUNE)
  val_ds = val_ds.map(strategy.squeeze_map, tf.data.AUTOTUNE)
  test_ds = test_ds.map(strategy.squeeze_map, tf.data.AUTOTUNE)
  # val_ds = val_ds.shard(num_shards=2, index=1)

  train_ds = train_ds.map(strategy.get_audio_feature_map, tf.data.AUTOTUNE)
  val_ds = val_ds.map(strategy.get_audio_feature_map, tf.data.AUTOTUNE)
  test_ds = test_ds.map(strategy.get_audio_feature_map, tf.data.AUTOTUNE)

  if save_af is True:
    train_ds = train_ds.map(strategy.save_audio_feature_map, tf.data.AUTOTUNE)

  train_ds = train_ds.map(strategy.reshape_map, tf.data.AUTOTUNE)
  val_ds = val_ds.map(strategy.reshape_map, tf.data.AUTOTUNE)
  test_ds = test_ds.map(strategy.reshape_map, tf.data.AUTOTUNE)

  # Training model
  train_ds = train_ds.cache().shuffle(
    10000).prefetch(tf.data.AUTOTUNE)
  val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

  for example, example_spect_labels in train_ds.take(1):
    input_shape = example.shape[1:]
    num_labels = len(label_names)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train_ds.map(
      map_func=lambda spec, label: spec))

    model = models.Sequential([
      layers.Input(shape=input_shape),
      # Downsample the input.
      layers.Resizing(32, 32),
      # Normalize.
      norm_layer,
      layers.Conv2D(32, 3, activation='relu'),
      layers.Conv2D(64, 3, activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.25),
      # # Flatten the result to feed into DNN
      layers.Flatten(),
      # 1st Dense layer with 512 neurons.
      layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
      layers.Dropout(0.25),
      # 2nd Dense layer with 128 neurons.
      layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
      layers.Dropout(0.25),
      # 3rd Dense layer with 64 neurons.
      layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
      layers.Dropout(0.5),
      layers.Dense(num_labels, activation='softmax'),
    ])
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
    )

    model.summary()

    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=EPOCHS,
      callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    print('Evaluation...')
    model.evaluate(test_ds, return_dict=True)

    #Save model
    print('Saving model...')
    export = ExportModel(model=model, label_names=label_names, input_shape=input_shape)
    model_dir = pathlib.Path(os.path.join(ASSETS_PATH, 'models', 'm_{}_{}'.format(DURATION, af_type_value)))
    tf.saved_model.save(export, model_dir)

    print('Model is saved to: {}'.format(model_dir))

    save_history(history, metrics=history.history, af_type=af_type)
    save_metrics(history.history, af_type=af_type)




