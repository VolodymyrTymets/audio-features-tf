from src.definitions import DURATION, labels, sub_sets
from src.data_set.data_set_splitter import DataSetSplitter


def start():

  splitter = DataSetSplitter(in_path='data_set_2000', out_path=f'data_set_{DURATION}', sub_sets=sub_sets, labels=labels)
  splitter.split(duration=DURATION)

if __name__ == "__main__":
  start()