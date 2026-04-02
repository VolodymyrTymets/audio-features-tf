from src.definitions import  labels, sub_sets
from src.data_set.data_set_fragmenter import DataSetFragmenter


def start():
  service = DataSetFragmenter(in_path='emg_data_set', out_path='emg_data_set_fragment', sub_sets=sub_sets, labels=labels)
  service.start()

if __name__ == "__main__":
  start()