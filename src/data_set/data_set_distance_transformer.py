import numpy as np
import random
from src.data_set.data_set_transformer import DataSetTransformer

class DataSetDistanceTransformer(DataSetTransformer):
  def __init__(self, in_path: str, out_path: str, sub_sets: list[str], labels: list[str]):
    super().__init__(in_path=in_path, out_path=out_path, sub_sets=sub_sets, labels=labels)

  def argument_distance(self, except_sets: list['str'] = [], except_labels: list[str] = []):
    self.logger.log('Start argumentation...')
    for signal, sr, set_name, label, path, file in self.read_data_set(log=False):
      # transformations are only for train set
      if set_name in except_sets:
        continue
      # transformations are only for train set
      if label in except_labels:
        continue


      xp = [1, 11]
      fp = [12, 24]

      for i in range(1, 11):
        if i == 0:
          continue
        distance_index = i

        # normalize_down_to = random.randint(i, i + 1) / 10
        # normalize_down_to = (i / 10) / 2
        normalize_down_to = np.interp(i, xp, fp)
        self.logger.log(f'Normalized down {distance_index} to {normalize_down_to}')
        normalized = self.transformer.normalize_down(signal, normalize_down_to)
        new_path = path.replace('rln', str(distance_index))
        self.logger.log(f'saved to {new_path}')
        self.write_signal(normalized, sr, new_path, f'd_{i}_')

    self.logger.log('End argumentation!')

