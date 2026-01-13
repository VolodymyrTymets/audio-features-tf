from typing import List
import tensorflow as tf
import numpy as np
from src.audio_features.audio_features import FrequencyDomainFeatures
from src.audio_features.types import AFTypes
from src.definitions import FRAGMENT_LENGTH
from src.neural_network.strategy.strategies.fft_strategy import FFTStrategy
from src.neural_network.strategy.strategies.stft_strategy import STFTStrategy
from src.neural_network.strategy.strategies.mel_strategy import MelStrategy
from src.neural_network.strategy.strategies.mfcc_strategy import MFCCStrategy
from src.neural_network.strategy.strategies.strategy_interface import IAFStrategy
from src.neural_network.strategy.train_strategy_interface import ITrainStrategy


class TrainStrategy(ITrainStrategy):
  def __init__(self, label_names: List[str], sr: int, frame_length: int, hop_length: int):
    self.features = FrequencyDomainFeatures()
    self.label_names = label_names
    self.sr = sr
    self.frame_length = frame_length
    self.hop_length = hop_length
    self.type = type
    self.stft_strategy = STFTStrategy(frame_length, hop_length)
    self.fft_strategy = FFTStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.mel_strategy = MelStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.mfcc_strategy = MFCCStrategy(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.strategy: IAFStrategy = self.stft_strategy
    self.shape = None

  def _calculate_shape(self):
    signal = np.zeros(FRAGMENT_LENGTH)
    af = self.strategy.get_audio_feature(signal)
    return af.shape

  def _get_audio_feature(self, waveform):
    bunch = waveform.numpy()
    if self.strategy is None:
      raise ValueError("Strategy not set")
    return np.array([self.strategy.get_audio_feature(signal) for signal in bunch])

  def _save_audio_feature(self, af, labels):
    bunch = af.numpy()
    if self.strategy is None:
      raise ValueError("Strategy not set")
    first = bunch[0]
    first_label_index = labels.numpy()[0]
    self.strategy.save_audio_feature(first, self.label_names[first_label_index])
    return af

  def set_strategy(self, strategy_type: AFTypes):
    if strategy_type.value == AFTypes.stft.value:
      self.strategy = self.stft_strategy
    elif strategy_type.value == AFTypes.fft.value:
      self.strategy = self.fft_strategy
    elif strategy_type.value == AFTypes.mel.value:
      self.strategy = self.mel_strategy
    elif strategy_type.value == AFTypes.mfcc.value:
      self.strategy = self.mfcc_strategy
    else:
      raise ValueError(f"Unknown strategy type: {strategy_type.value}")
    self.shape = self._calculate_shape()


  @tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
  def get_audio_feature(self, i):
    return tf.py_function(self._get_audio_feature, [i], tf.float32)

  @tf.function
  def save_audio_feature(self, i, labels):
    tf.py_function(self._save_audio_feature, [i, labels], tf.float32)
    return i

  @tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
  def reshape(self, i):
    w, h = self.shape
    return tf.reshape(i, (-1, w, h, 1))

  # map function for tf.data.Datasets
  def get_audio_feature_map(self, audio, labels):
    audio = self.get_audio_feature(audio)
    return audio, labels

  def save_audio_feature_map(self, audio, labels):
    audio = self.save_audio_feature(audio, labels)
    return audio, labels

  def reshape_map(self, audio, labels):
    audio = self.reshape(audio)
    return audio, labels

  def squeeze_map(self, audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels
