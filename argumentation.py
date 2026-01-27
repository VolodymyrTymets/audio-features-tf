from src.definitions import DURATION
from src.neural_network.augmentation.augmentation_pipline import AugmentationPipline


def start():
  augmentation_pipline = AugmentationPipline('data_set_2000_test', 'data_set_0.2', labels=['stimulation'])
  augmentation_pipline.start(duration=DURATION)

if __name__ == "__main__":
  start()