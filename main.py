import librosa
from src.audio_features.types import AFTypes
from src.neural_network.train import train
from src.neural_network.valid_record import valid_record
# from src.neural_network.filter_data_set import filter_data_set

def main():
  n_mels_list = [8, 16, 24, 32, 48, 64, 96, 128, 256]
  for n_mels in n_mels_list:
    print("Starting training...")
    train(AFTypes.mel, n_mels=n_mels, save_af=True)
    print("Training finished.")

    print("Starting validation of record...")
    valid_record(AFTypes.mel, n_mels=n_mels)
    print("End validation of record.")

if __name__ == "__main__":
  main()