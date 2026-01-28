from src.definitions import DURATION, labels, sub_sets
from src.data_set.data_set_transformer import DataSetTransformer


def start():
  dataset_name = f'data_set_{DURATION}'
  service = DataSetTransformer(in_path=dataset_name, out_path=dataset_name, sub_sets=sub_sets, labels=labels)
  service.normalise(except_sets=['test'], except_labels=['noise', 'breath'])

if __name__ == "__main__":
  start()