
from src.definitions import DURATION, labels, sub_sets
from src.data_set.data_set_distance_transformer import DataSetDistanceTransformer
from src.data_set.data_set_record_generator import DataSetRecordGenerator


def start():
  dataset_name = f'data_set_2000'
  except_labels = labels.copy()
  except_labels.remove('rln')
  service = DataSetDistanceTransformer(in_path=dataset_name, out_path=dataset_name, sub_sets=sub_sets, labels=labels)
  record_generator = DataSetRecordGenerator(in_path=dataset_name, out_path=dataset_name, sub_sets=sub_sets, labels=labels)
  # service.argument_distance(except_sets=[], except_labels=except_labels)
  record_generator.generate_test_record(except_sets=['train'])



if __name__ == "__main__":
  start()
