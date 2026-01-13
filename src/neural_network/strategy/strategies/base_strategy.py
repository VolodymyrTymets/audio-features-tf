import os
import numpy as np
import matplotlib.pyplot as plt
import uuid
from PIL import Image
from src.audio_features.audio_features import FrequencyDomainFeatures
from src.definitions import ASSETS_PATH
from src.neural_network.strategy.strategies.strategy_interface import IAFStrategy
from src.files import Files

class BaseStrategy(IAFStrategy):
  def __init__(self, sr: int, frame_length: int, hop_length: int):
    self.features = FrequencyDomainFeatures()
    self.frame_length = frame_length
    self.hop_length = hop_length
    self.sr = sr
    self.files = Files()

  def _get_image_path(self, label: str):
    file_name = f"{self.af_type.value}_{uuid.uuid4()}.png"
    directory = self.files.join(ASSETS_PATH, '__af__', self.af_type.value, label)
    self.files.create_folder(directory)
    return os.path.join(directory, file_name)

  def save_audio_feature(self, matrix: np.ndarray, label: str):
    file_path = self._get_image_path(label=label)
    normalized = matrix.astype(np.uint8)

    # Create image and save
    img = Image.fromarray(normalized)
    img.save(file_path)
    return matrix

  def get_audio_feature(self, wave: np.ndarray):
    pass