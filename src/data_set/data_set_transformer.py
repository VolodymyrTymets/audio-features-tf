import random
from src.data_set.data_set_file_worker import DataSetFileWorker
from src.logger.logger_service import Logger
from src.audio_features.signal_transformer import SignalTransformer

from src.definitions import  sr as SR, frame_length, hop_length


class DataSetTransformer(DataSetFileWorker):
  def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str]):
    super().__init__(in_path=in_path, out_path=out_path, sub_sets=sub_sets, labels=labels)
    self.transformer = SignalTransformer(sr=SR, frame_length=frame_length, hop_length=hop_length)
    self.logger = Logger('AugmentationPipline')

  def normalise(self, except_sets: list['str'] = [] , except_labels: list[str] = []):
    self.logger.log('Start normalisation')
    for signal, sr, set_name, label, path, file in self.read_data_set(log=False):
      # transformations are only for train set
      if set_name in except_sets:
        continue
      # transformations are only for train set
      if label in except_labels:
        continue
      normalize_fragments = self.transformer.normalize(signal)
      for normalize_index, n_fragment in enumerate(normalize_fragments):
        self.write_signal(n_fragment, sr, path, f'norm_{normalize_index}')
    self.logger.log('End normalisation')


