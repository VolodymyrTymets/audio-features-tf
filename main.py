from src.audio_features.types import AFTypes
from src.neural_network.train import train
from src.neural_network.valid import valid
from src.neural_network.valid_record import valid_record

def main():
  audio_futures = [AFTypes.fft, AFTypes.stft]
  for af_type in audio_futures:
    print("Starting training...")
    train(af_type, show_plot=False)
    print("Training finished.")
    print("Starting validation...")
    valid(af_type, show_plot=False)
    print("End validation.")
    print("Starting validation of record...")
    valid_record(af_type, show_plot=True)
    print("End validation of record.")

if __name__ == "__main__":
  main()