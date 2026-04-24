from src.definitions import DURATION, labels, sub_sets, dataset_name
from src.data_set.data_set_splitter import DataSetSplitter

name = dataset_name.replace(f'_{DURATION}', '')
def start():
  splitter = DataSetSplitter(in_path=name, out_path=dataset_name, sub_sets=sub_sets, labels=labels)
  splitter.split(duration=DURATION)

if __name__ == "__main__":
  start()