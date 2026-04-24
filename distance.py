
from src.definitions import labels, sub_sets, distance_points, dataset_name, DURATION
from src.data_set.data_set_distance_transformer import DataSetDistanceTransformer

ds_name = dataset_name.replace(f'_{DURATION}', '')
def start():
  except_labels = labels.copy() + distance_points.copy()
  except_labels.remove('stimulation')
  service = DataSetDistanceTransformer(in_path=ds_name, out_path=ds_name, sub_sets=sub_sets, labels=labels)
  service.argument_distance(except_sets=[], except_labels=except_labels, normalize_rage=[10, 24])


if __name__ == "__main__":
  start()
