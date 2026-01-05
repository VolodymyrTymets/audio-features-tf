import pathlib
import os
import numpy as np
import uuid
from PIL import Image
from src.audio_features.audio_features import FrequencyDomainFeatures
from src.definitions import ASSETS_PATH
from src.neural_network.strategy.strategies.strategy_interface import IAFStrategy
from src.audio_features.types import AFTypes
from src.files import Files


class STFTStrategy(IAFStrategy):
  def __init__(self, frame_length: int, hop_length: int):
    self.features = FrequencyDomainFeatures()
    self.frame_length = frame_length
    self.hop_length = hop_length
    self.files = Files()

  def _get_image_path(self, label: str):
    file_name = f"stft_{uuid.uuid4()}.png"
    directory = self.files.join(ASSETS_PATH, '__af__', AFTypes.stft.value, label)
    self.files.create_folder(directory)
    return os.path.join(directory, file_name)

  def get_audio_feature(self, wave: np.ndarray):
    return self.features.stft(signal=wave, frame_length=self.frame_length, hop_length=self.hop_length)

  def save_audio_feature(self, stft: np.ndarray, label: str):
    file_path = self._get_image_path(label=label)
    # Normalize to 0-255 range
    # if stft.max() > 0:
    #   normalized = (stft / stft.max() * 255).astype(np.uint8)
    # else:
    normalized = stft.astype(np.uint8)

    # Create image and save
    img = Image.fromarray(normalized)
    img.save(file_path)
    return stft
