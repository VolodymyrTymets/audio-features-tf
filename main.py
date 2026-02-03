from src.audio_features.types import AFTypes
from src.neural_network.model.types import ModelTypes
from src.neural_network.MLPipline import MLPipeline
from src.neural_network.valid_record import valid_record

def main():
  mpt = MLPipeline(af_type=AFTypes.mel, model_type=ModelTypes.CNN)
  mpt.train(save_af=False)
  valid_record(af_type=AFTypes.mel, model_type=ModelTypes.CNN)

if __name__ == "__main__":
  main()