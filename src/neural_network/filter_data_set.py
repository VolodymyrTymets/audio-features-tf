import shutil
import librosa
import wave
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.audio_features.types import AFTypes
from src.definitions import DURATION, FRAGMENT_LENGTH
from src.files import Files

files = Files()
nFFT = 512

def buffer_to_float_32(buffer):
  # Convert buffer to float32 using NumPy
  audio_as_np_int16 = np.frombuffer(buffer, dtype=np.int16)
  audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

  # Normalise float32 array so that values are between -1.0 and +1.0
  max_int16 = 2 ** 15
  audio_normalised = audio_as_np_float32 / max_int16
  return audio_normalised

def get_wave(file_full_path):
  wav_file = wave.open(file_full_path, 'rb')
  data = wav_file.readframes(nFFT)
  chunks = []
  while data != b'':
    chunk = buffer_to_float_32(data)
    chunks = np.concatenate((chunks, chunk))
    data = wav_file.readframes(nFFT)
  return chunks

def get_chunk_label_by_model(wave, model):
  x = tf.convert_to_tensor(wave, dtype=tf.float32)
  waveform = x[tf.newaxis,...]
  result = model(tf.constant(waveform))
  label_names = np.array(result['label_names'])
  label_names = label_names.astype(str)
  prediction = tf.nn.softmax(result['predictions']).numpy()[0]
  max_value = max(prediction)
  i, = np.where(prediction == max_value)
  wave_label = label_names[i]
  return wave_label

def filter_data_set(af_type: AFTypes, data_set_name: str):

  model_dir = files.join(files.ASSETS_PATH, 'models', f'm_{DURATION}_{af_type.value}')
  model = tf.saved_model.load(model_dir)

  sets = ['train', 'valid']
  labels = ['noise', 'breath', 'stimulation']
  for set_name in sets:
    for label in labels:
      dats_set_path = files.join(files.ASSETS_PATH, data_set_name, set_name)
      dir_path = files.join(dats_set_path, label)
      files_path = files.get_only_files(dir_path)

      for file in files_path:
        wave = get_wave(files.join(dir_path, file))
        if len(wave) >= FRAGMENT_LENGTH:
          wave_label = get_chunk_label_by_model(wave=wave[:FRAGMENT_LENGTH], model=model)[0]
          if (label != wave_label):
            from_path = files.join(dats_set_path, label, file)
            out_folder = files.join(dats_set_path, '__filtered__', wave_label)
            files.create_folder(out_folder)
            to_path = files.join(out_folder, file)
            # print('{} -> {}'.format(from_path, to_path))
            shutil.move(from_path, to_path)


