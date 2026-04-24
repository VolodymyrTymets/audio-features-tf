from src.audio_features.types import AFTypes
from src.files import Files
from src.neural_network.model.types import ModelTypes
from src.neural_network.MLPipline import MLPipeline
from src.definitions import dataset_name


file = Files()
def main():

  af_types = [AFTypes.mfcc]
  model_types = [ModelTypes.CUSTOM]
  for af_type in af_types:
    for model_type in model_types:
      mpt = MLPipeline(af_type=af_type, model_type=model_type)
      mpt.evaluate_records(data_set_name=dataset_name)
      mpt.label_records(data_set_name=dataset_name)

if __name__ == "__main__":
  main()