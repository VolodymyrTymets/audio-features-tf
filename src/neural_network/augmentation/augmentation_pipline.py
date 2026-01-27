import uuid

import soundfile as sf
import librosa
import numpy as np

from src.logger.logger_service import Logger
from src.neural_network.augmentation.data_transformer import DataTransformer
from src.files import Files
from src.definitions import ASSETS_PATH, sr, frame_length, hop_length


class AugmentationPipline:
  def __init__(self, in_path: str, out_path: str, labels: list[str]):
    self.in_path = in_path
    self.out_path = out_path
    self.labels = labels

    self.transformer = DataTransformer(sr=sr, frame_length=frame_length, hop_length=hop_length)
    self.files = Files()
    self.logger = Logger('AugmentationPipline')
    self.set_names = ['train', 'test']

    # self.files.remove(self.files.join(ASSETS_PATH, self.out_path))
    self.files.create_folder(self.files.join(ASSETS_PATH, self.out_path))

  def _write(self, signal: np.ndarray, path: str, sr: int, prefix: str):
    out_path = path.replace(self.in_path, self.out_path)
    self.files.create_folder(out_path)
    file_name = self.files.join(out_path, f'{prefix}_' + uuid.uuid4().hex + '.wav')
    self.logger.log(f'--> write to: {file_name}')
    sf.write(file_name, signal, samplerate=sr)

  def argument(self, signal: np, duration: float, in_path: str, set_name: str = 'train'):
    fragments = self.transformer.split(signal=signal, duration=duration)

    for time_index, fragment in enumerate(fragments):
      self._write(fragment, in_path, sr, str(time_index))
      # all next transformations are only for train set
      if set_name != 'test':
        normalize_fragments = self.transformer.normalize(fragment)
        for normalize_index, n_fragment in enumerate(normalize_fragments):
          self._write(n_fragment, in_path, sr, f'{time_index}_norm_{normalize_index}')

    return fragments

  def start(self, duration: float):
    for set_name in self.set_names:
      for label in self.labels:
        path = self.files.join(ASSETS_PATH, self.in_path, set_name, label)
        files = self.files.get_only_files(path)

        for file in files:
          file_path = self.files.join(path, file)
          self.logger.log(f'--> read from: {file_path}', 'blue')
          signal, sr = librosa.load(file_path)

          self.argument(signal, duration, path, set_name)
