from src.audio_features.types import AFTypes
from src.definitions import DURATION, labels, sub_sets
from src.data_set.data_set_filter import DataSetFilter


def start():
  dataset_name = f'data_set_{DURATION}'
  strategy_type = AFTypes.mel
  service = DataSetFilter(in_path=dataset_name, out_path=dataset_name, sub_sets=sub_sets, labels=labels)
  service.filter(except_sets=[], except_labels=['noise'], model_name=f'm_{DURATION}_{strategy_type.value}', strategy_type=strategy_type)

if __name__ == "__main__":
  start()