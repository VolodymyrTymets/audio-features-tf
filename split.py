from src.definitions import DURATION, labels, sub_sets
from src.data_set.data_set_splitter import DataSetSplitter

name = 'data_set_ecg_distance'
def start():
  splitter = DataSetSplitter(in_path=name, out_path=f'{name}_{DURATION}', sub_sets=sub_sets, labels=labels)
  splitter.split(duration=DURATION)

if __name__ == "__main__":
  start()