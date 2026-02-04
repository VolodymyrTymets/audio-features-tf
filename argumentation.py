from src.audio_features.types import ArgumentationTypes
from src.definitions import DURATION, labels, sub_sets
from src.data_set.data_set_transformer import DataSetTransformer


def start():
  dataset_name = f'data_set_{DURATION}'
  except_sets = ['test']
  except_labels = ['noise']
  service = DataSetTransformer(in_path=dataset_name, out_path=dataset_name, sub_sets=sub_sets, labels=labels)
  service.argument(argumentation_types=[
    ArgumentationTypes.time_stretch.value,
    ArgumentationTypes.normalization.value,
    ArgumentationTypes.time_shift.value,
    ArgumentationTypes.pitch_shift.value
  ], except_sets=except_sets, except_labels=except_labels)


if __name__ == "__main__":
  start()
