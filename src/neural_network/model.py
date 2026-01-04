import tensorflow as tf
import numpy as np
from src.audio_features.audio_features import FrequencyDomainFeatures
from src.definitions import sr, FRAGMENT_LENGTH, frame_length, hop_length


features = FrequencyDomainFeatures()

def _get_audio_feature_shape(waveform):
    spectrogram = features.stft(signal=waveform.numpy(), frame_length=frame_length, hop_length=hop_length)
    return spectrogram.shape

def _get_audio_feature(waveform):
    spectrogram = features.stft(signal=waveform.numpy(), frame_length=frame_length, hop_length=hop_length)
    # spectrogram = spectrogram[0, ...]
    spectrogram = np.moveaxis(spectrogram, 0, -1)
    # spectrogram = spectrogram[..., tf.newaxis]
    print('-----> spectrogram', spectrogram.shape)

    return spectrogram


# clean up magical data
@tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
def get_audio_feature(i):
  return tf.py_function(_get_audio_feature, [i], tf.float32)

@tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
def reshape(i):
  # todo: remove hardcoded values
  return tf.reshape(i, (-1, 513, 35, 1))

# @tf.function(input_signature=[tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32)])
# # @tf.py_function(Tout=[tf.TensorSpec(shape=[35, 513, 1], dtype=tf.float32)])
# def _get_spectrogram(waveform):
#     # Convert the waveform to a spectrogram via a STFT.
#     spectrogram = tf.signal.stft(
#       waveform, frame_length=frame_length, frame_step=hop_length)
#     # Obtain the magnitude of the STFT.
#     spectrogram = tf.abs(spectrogram)
#     # Add a `channels` dimension, so that the spectrogram can be used
#     # as image-like input data with convolution layers (which expect
#     # shape (`batch_size`, `height`, `width`, `channels`).
#     spectrogram = spectrogram[..., tf.newaxis]
#     return spectrogram
#     # spectrogram = features.stft(signal=waveform.numpy(), frame_length=frame_length, hop_length=hop_length)
#     # spectrogram = spectrogram[0, ...]
#     # spectrogram = np.moveaxis(spectrogram, 0, -1)
#     # spectrogram = spectrogram[..., tf.newaxis]
#
#     #return spectrogram


class ExportModel(tf.Module):
    def __init__(self, model, label_names, fragment_length = FRAGMENT_LENGTH):
        self.model = model
        self.label_names = label_names
        self.fragment_length = fragment_length

        self.__call__.get_concrete_function(
            x=tf.TensorSpec(shape=[None, FRAGMENT_LENGTH], dtype=tf.float32))

    @tf.function
    def __call__(self, x):
        x = get_audio_feature(x)
        x = reshape(x)
        result = self.model(x, training=False)

        class_ids = tf.argmax(result, axis=-1)
        class_names = tf.gather(self.label_names, class_ids)
        return {'predictions': result,
                'class_ids': class_ids,
                'class_names': class_names,
                'label_names': self.label_names}
    