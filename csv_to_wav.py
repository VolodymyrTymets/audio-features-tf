
from src.data_set.data_set_csv_to_wav import DataSetCSVToWav


def start():
  dataset_name = f'data_set'
  service = DataSetCSVToWav(in_path=dataset_name, out_path=dataset_name, sub_sets=['train'], labels=['hand_movements'])
  service.csv_to_wav()



if __name__ == "__main__":
  start()
