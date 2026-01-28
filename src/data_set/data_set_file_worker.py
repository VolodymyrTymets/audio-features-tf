import uuid

import soundfile as sf
import librosa
import numpy as np
from src.files import Files
from src.logger.logger_service import Logger
from src.definitions import ASSETS_PATH

class DataSetFileWorker:
  def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str]):
    self.in_path = in_path
    self.out_path = out_path
    self.labels = labels
    self.set_names = sub_sets
    self.files = Files()
    self.logger = Logger('DataSetFileWorker')

  def write_signal(self, signal: np.ndarray, sr: int, file_path: str, prefix: str):
    out_path = file_path.replace(self.in_path, self.out_path)
    self.files.create_folder(out_path)
    file_name = self.files.join(out_path, f'{prefix}_' + uuid.uuid4().hex + '.wav')
    self.logger.log(f'--> write to: {file_name}')
    sf.write(file_name, signal, samplerate=sr)

  def read_data_set(self, log: bool = True):
    for set_name in self.set_names:
      for label in self.labels:
        path = self.files.join(ASSETS_PATH, self.in_path, set_name, label)
        files = self.files.get_only_files(path)

        for file in files:
          file_path = self.files.join(path, file)
          if log:
            self.logger.log(f'--> read from: {file_path}', 'blue')
          signal, sr = librosa.load(file_path)

          yield signal, sr, set_name, label, path, file