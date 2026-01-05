import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import uuid
from PIL import Image
from src.audio_features.audio_features import FrequencyDomainFeatures
from src.definitions import ASSETS_PATH
from src.neural_network.strategy.strategies.strategy_interface import IAFStrategy
from src.audio_features.types import AFTypes
from src.files import Files

matplotlib.use('Agg')

class FFTStrategy(IAFStrategy):
  def __init__(self, sr: int, frame_length: int, hop_length: int):
    self.features = FrequencyDomainFeatures()
    self.frame_length = frame_length
    self.hop_length = hop_length
    self.sr = sr
    self.files = Files()

    width_pixels = 128
    height_pixels = 128
    dpi_value = 100  # A common default or desired DPI

    # # Calculate the size in inches: inches = pixels / dpi
    # width_inches = width_pixels / dpi_value
    # height_inches = height_pixels / dpi_value
    #
    # # Create the figure with the specified size and DPI
    # # plt.figure(figsize=(width_inches, height_inches), dpi=dpi_value)
    fig, ax = plt.subplots()
    self.fig = fig
    self.ax = ax
    # self.fig.figsize = (width_inches, height_inches)

  def _get_image_path(self, label: str):
    file_name = f"fft_{uuid.uuid4()}.png"
    directory = self.files.join(ASSETS_PATH, '__af__', AFTypes.fft.value, label)
    self.files.create_folder(directory)
    return os.path.join(directory, file_name)

  def get_audio_feature(self, wave: np.ndarray):
    # mag, freq =  self.features.fft(signal=wave, sr=self.sr, frame_length=self.frame_length, hop_length=self.hop_length)

    # self.ax.plot(freq, mag)
    # # Draw the canvas to ensure it's rendered
    # self.fig.canvas.draw()
    # image_flat = np.asarray(self.fig.canvas.buffer_rgba())
    # image_flat = image_flat[..., -1]
    # return image_flat

    matrix, freqs, bins, im = self.ax.specgram(wave, Fs=self.sr, NFFT=self.frame_length, cmap='plasma')
    # print(im.shape)
    return matrix

  def save_audio_feature(self, stft: np.ndarray, label: str):
    file_path = self._get_image_path(label=label)
    normalized = stft.astype(np.uint8)

    # Create image and save
    img = Image.fromarray(normalized)
    img.save(file_path)
    return stft
