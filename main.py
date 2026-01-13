import librosa
from src.audio_features.types import AFTypes
from src.neural_network.train import train
from src.neural_network.valid_record import valid_record
# from src.neural_network.filter_data_set import filter_data_set

def main():
  audio_features = [AFTypes.mfcc, AFTypes.mel]
  for af_type in audio_features:
    print("Starting training...")
    train(af_type, save_af=True)
    print("Training finished.")

    print("Starting validation of record...")
    valid_record(af_type, show_plot=True)
    print("End validation of record.")

    # filter_data_set(af_type, data_set_name='data_set_0.2_filtered')

if __name__ == "__main__":
  main()