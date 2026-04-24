
from src.definitions import labels, sub_sets, dataset_name
from src.data_set.data_set_record_generator import DataSetRecordGenerator

COUNT = 2

def start(count: int):
  for i in range(count):
    record_generator = DataSetRecordGenerator(in_path=dataset_name, out_path=dataset_name,
                                              sub_sets=sub_sets, labels=labels)
    record_generator.generate_test_record(except_sets=['train'], except_labels=['ecg', 'wrist_extension'])

if __name__ == "__main__":
  start(COUNT)
