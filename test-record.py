
from src.definitions import labels, sub_sets
from src.data_set.data_set_record_generator import DataSetRecordGenerator

COUNT = 2

dataset_name = f'data_set_ecg_distance_0.05'
output_path = './assets/test/records'

def start(count: int):
  for i in range(count):
    record_generator = DataSetRecordGenerator(in_path=dataset_name, out_path=dataset_name,
                                              sub_sets=sub_sets, labels=labels)
    record_generator.generate_test_record(except_sets=['train'], except_labels=['ecg', 'wrist_extension'], out_path=output_path)

if __name__ == "__main__":
  start(COUNT)
