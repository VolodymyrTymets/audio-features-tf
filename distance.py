
from src.definitions import DURATION, labels, sub_sets, distance_points
from src.data_set.data_set_distance_transformer import DataSetDistanceTransformer
from src.data_set.data_set_record_generator import DataSetRecordGenerator


def start():
  dataset_name = f'data_set_ecg_distance'
  except_labels = labels.copy() + distance_points.copy()
  except_labels.remove('stimulation')
  # service = DataSetDistanceTransformer(in_path=dataset_name, out_path=dataset_name, sub_sets=sub_sets, labels=labels)
  # service.argument_distance(except_sets=[], except_labels=except_labels, normalize_rage=[10, 24])

  dataset_name = f'data_set_ecg_distance_0.05'
  record_generator = DataSetRecordGenerator(in_path=dataset_name, out_path=dataset_name,
                                            sub_sets=sub_sets, labels=labels)
  record_generator.generate_test_record(except_sets=['train'], except_labels=['ecg', 'wrist_extension'])



if __name__ == "__main__":
  start()
