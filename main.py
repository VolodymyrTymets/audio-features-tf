from src.audio_features.types import AFTypes
from src.neural_network.train import train

def main():
  print("Starting training...")
  train(AFTypes.stft)
  print("Training finished.")

if __name__ == "__main__":
  main()