from src.audio_features.types import AFTypes
from src.neural_network.model.types import ModelTypes
from src.neural_network.train import train
from src.neural_network.valid_record import valid_record

def main():
  train(af_type=AFTypes.mel, model_type=ModelTypes.CNN ,  save_af=False)
  valid_record(af_type=AFTypes.mel, show_plot=True)

if __name__ == "__main__":
  main()