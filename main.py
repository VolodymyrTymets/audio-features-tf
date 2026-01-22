import librosa
from src.audio_features.types import AFTypes
from src.neural_network.train import train
from src.neural_network.valid_record import valid_record

def main():
  audio_features = [AFTypes.wave, AFTypes.fft, AFTypes.stft, AFTypes.mfcc, AFTypes.mel, AFTypes.bw, AFTypes.sc, AFTypes.ae, AFTypes.rms, AFTypes.zcr, AFTypes.ber]
  for af_type in audio_features:
    print("Starting training...")
    train(af_type, save_af=False)
    print("Training finished.")

    print("Starting validation of record...")
    valid_record(af_type, show_plot=True)
    print("End validation of record.")

if __name__ == "__main__":
  main()