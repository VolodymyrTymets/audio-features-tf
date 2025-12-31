import os
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from matplotlib import pyplot as plt

from src.neural_network.model import ExportModel, get_spectrogram
from src.definitions import DURATION, FRAGMENT_LENGTH, EPOCHS, ASSETS_PATH

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels


def ds_to_spectrogram(ds):
  return ds.map(
    map_func=lambda audio, label: (get_spectrogram(audio), label),
    num_parallel_calls=tf.data.AUTOTUNE)


def train(show_plot=False):
  print('Training model...', tf.executing_eagerly())
  # Form data storage
  data_dir = pathlib.Path(os.path.join(ASSETS_PATH, 'data_set_{}'.format(DURATION), 'train'))
  train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=32,
    validation_split=0.2,
    seed=0,
    output_sequence_length=FRAGMENT_LENGTH,
    subset='both')
  label_names = np.array(train_ds.class_names)

  # Prepare data - wave to spectrogram
  train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
  val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
  val_ds = val_ds.shard(num_shards=2, index=1)

  train_spectrogram_ds = ds_to_spectrogram(train_ds)
  val_spectrogram_ds = ds_to_spectrogram(train_ds)

  # Training model
  train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(
    10000).prefetch(tf.data.AUTOTUNE)
  val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

  history = None
  for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    input_shape = example_spectrograms.shape[1:]
    num_labels = len(label_names)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train_spectrogram_ds.map(
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
      # Flatten the result to feed into DNN
      layers.Dropout(0.25),
      layers.Flatten(),
      layers.Dense(128, activation='tanh'),
      layers.Dropout(0.5),
      layers.Dense(num_labels, activation='softmax'),
    ])

    model.summary()

    model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'],
      # run_eagerly=True
    )

    history = model.fit(
      train_spectrogram_ds,
      validation_data=val_spectrogram_ds,
      epochs=EPOCHS,
      callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )


    # Save model
    export = ExportModel(model=model, label_names=label_names, fragment_length=FRAGMENT_LENGTH)
    model_dir = pathlib.Path(os.path.join(ASSETS_PATH, 'models', 'm_{}_{}'.format(DURATION, FRAGMENT_LENGTH)))
    tf.saved_model.save(export, model_dir)

    print('Model is saved to: {}'.format(model_dir))
  if show_plot:
    metrics = history.history
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

    plt.show()
