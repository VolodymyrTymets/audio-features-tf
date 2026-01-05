from src.audio_features.types import AFTypes
from src.neural_network.train import train
from src.neural_network.valid import valid
from src.neural_network.valid_record import valid_record

def main():
  print("Starting training...")
  train(AFTypes.stft, show_plot=False)
  print("Training finished.")
  print("Starting validation...")
  valid(AFTypes.stft, show_plot=True)
  print("End validation.")
  print("Starting validation of record...")
  valid_record(AFTypes.stft, show_plot=True)
  print("End validation of record.")

if __name__ == "__main__":
  main()