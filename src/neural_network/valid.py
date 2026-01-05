import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.audio_features.types import AFTypes
from src.definitions import DURATION, FRAGMENT_LENGTH
from src.files import Files

files = Files()

def to_chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    yield lst[i:i + n]

def get_wave(file_full_path):
  waveform, sr = librosa.load(file_full_path)
  chunks = [x for x in to_chunks(waveform, int(FRAGMENT_LENGTH))]
  return chunks

def valid(af_type: AFTypes, show_plot=False):
  model_dir = files.join(files.ASSETS_PATH, 'models', f'm_{DURATION}_{af_type.value}')
  model = tf.saved_model.load(model_dir)

  valid_dir_path = files.join(files.ASSETS_PATH, 'test')
  labels = ['noise', 'breath', 'stimulation']
  label_colors = ['blue', 'green', 'red']
  predictions_by_label = []
  for label in labels:
    dir_path = files.join(valid_dir_path, label)
    files_path = files.get_only_files(dir_path)
    names = []
    waves = []
    predictions = []
    for file in files_path:
      chunks = get_wave(files.join(dir_path, file))
      waves.append(chunks)
      names.append(file)

    for chunks in waves:
      chunk_prediction = []
      for wave in chunks:
        try:
          x = tf.convert_to_tensor(wave, dtype=tf.float32)
          waveform = x[tf.newaxis, ...]
          result = model(tf.constant(waveform))
          prediction = result['predictions']
          chunk_prediction.append(tf.nn.softmax(prediction[0]).numpy()[0])
        except Exception as e:
          print('ERROR:', e)
          chunk_prediction.append(0)
      predictions.append(np.mean(chunk_prediction))
    predictions_by_label.append(predictions)


  mean_prediction_by_label =  [np.mean(prediction) * 100 for prediction in predictions_by_label]
  for i, prediction in enumerate(mean_prediction_by_label):
    print(labels[i], '->', prediction)
  total = np.mean(predictions_by_label)
  print('total ->', total)

  if show_plot == True:
    plt.figure(figsize=(12,2))
    plt.subplot(1,2,1)
    for i,prediction in enumerate(predictions_by_label):
      plt.plot(np.arange(len(prediction)) + 1, np.multiply(prediction, 100), color=label_colors[i])
    plt.legend(['n', 'b', 's'])
    plt.ylim([0, max(plt.ylim())])
    plt.ylabel('Prediciont %')
    plt.xlabel('Count of fragments')

    ax = plt.subplot(1,2,2)
    p = ax.bar(['n', 'b', 's', 'mean'], mean_prediction_by_label + [total])
    ax.bar_label(p, label_type='center')
    plt.ylabel('Prediciont %')
    plt.xlabel('Labels')
    plt.legend()

    plt.show()

  # cut_ext = np.vectorize(lambda f: f.replace('.wav', ''))
  # to_perc = np.vectorize(lambda x: x * 100)

  # fig, (ax_n, ax_b, ax_s) = plt.subplots(1, 3)
  # ax_X = range(len(n_prediction))
  # ax_n.bar(cut_ext(n_files), to_perc(n_prediction))
  # ax_n.set_ylabel('Prediction (%)', fontweight ='bold')
  # ax_n.set_xlabel('File (.wav)', fontweight ='bold')
  # ax_n.set_title('noise {}%'.format(round(total_n, 2)))

  # ax_b.bar(cut_ext(b_files), to_perc(b_prediction))
  # ax_b.set_xlabel('File (.wav)', fontweight ='bold')
  # ax_b.set_title('breat {}%'.format(round(total_b, 2)))

  # ax_s.bar(cut_ext(s_files), to_perc(s_prediction))
  # ax_s.set_xlabel('File (.wav)', fontweight ='bold')
  # ax_s.set_title('stimulation {}%'.format(round(total_s, 2)))


  # plt.show()

