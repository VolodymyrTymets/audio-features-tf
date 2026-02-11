from src.audio_features.types import AFTypes
from src.files import Files
from src.neural_network.model.types import ModelTypes
from src.neural_network.MLPipline import MLPipeline

file = Files()
def main():

  af_types = [AFTypes.mel, AFTypes.mfcc]
  model_types = [ModelTypes.CNN, ModelTypes.LSTM, ModelTypes.GRU]
  for af_type in af_types:
    for model_type in model_types:
      print(f'Training {af_type.value} with {model_type.value}')
      # if file.is_exist(file.join(file.ASSETS_PATH, '__af__', f'{af_type.value}_{model_type.value}')):
      #   continue
      mpt = MLPipeline(af_type=af_type, model_type=model_type)
      mpt.train(save_af=False)
      mpt.label_records()

  # mpt = MLPipeline(af_type=AFTypes.mel, model_type=ModelTypes.CNN)
  # mpt.train(save_af=False)
  # mpt.label_records()



if __name__ == "__main__":
  main()