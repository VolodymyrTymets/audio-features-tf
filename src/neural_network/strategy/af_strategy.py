import tensorflow as tf
import numpy as np
from src.audio_features.audio_features import FrequencyDomainFeatures
from src.definitions import FRAGMENT_LENGTH


class AF_Stratedgy:
  def __init__(self, sr: int, frame_length: int, hop_length: int):
    self.features = FrequencyDomainFeatures()
    self.sr = sr
    self.frame_length = frame_length
    self.hop_length = hop_length

  # todo: refactor to strategies
  def _get_audio_feature(self, waveform):
    signal = waveform.numpy()
    spectrogram = self.features.stft(signal=signal, frame_length=self.frame_length, hop_length=self.hop_length)
    spectrogram = np.moveaxis(spectrogram, 0, -1)
    return spectrogram


  @tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
  def get_audio_feature(self, i):
    return tf.py_function(self._get_audio_feature, [i], tf.float32)

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
  def reshape(self, i):
    # todo: remove hardcoded values
    return tf.reshape(i, (-1, 513, 35, 1))

  # map function for tf.data.Datasets
  def get_audio_feature_map(self, audio, labels):
    audio = self.get_audio_feature(audio)
    return audio, labels

  def reshape_map(self, audio, labels):
    audio = self.reshape(audio)
    return audio, labels

  def squeeze_map(self, audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels
