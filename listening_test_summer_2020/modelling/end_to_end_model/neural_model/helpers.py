# Lint as: python3
"""Callbacks for visualizing input images and predictions of loudness model."""
from typing import List

import tensorflow.compat.v2 as tf


class StepIncrementingCallback(tf.keras.callbacks.Callback):

  def on_train_batch_end(self, batch, logs=None):
    self.model.step.assign_add(1)


class ImageWriterCallback(tf.keras.callbacks.Callback):
  """Makes images of predictions by the model of a few selected examples."""

  def __init__(self,
               log_dir: str,
               channels: int,
               bins: int,
               ex_img_batch: List[tf.Tensor],
               sample_rate: int):
    super(ImageWriterCallback, self).__init__()
    self.file_writer = tf.summary.create_file_writer(log_dir)
    self.channels = channels
    self.bins = bins
    self.sample_rate = sample_rate
    self.hz_per_bin = float(self.sample_rate) * 0.5 / float(self.bins)
    self.ex_img_batch = ex_img_batch
    self.thickness = 16
    self._create_input_images()
    self._add_input_images()

  def on_train_batch_end(self, batch: int, logs=None):
    if batch % 200 == 0:
      self._add_output_images()

  def _compute_rgb_image(self, input_image: tf.Tensor) -> tf.Tensor:
    input_img = input_image
    signal_img = input_image[:, :, :, 0:1]
    noise_img = input_image[:, :, :, 1:2]
    snr_img = signal_img - noise_img
    power_img = tf.math.log(tf.pow(10, signal_img) +
                            tf.pow(10, noise_img)) / tf.math.log(10.0)
    power_img = tf.where(tf.math.is_inf(power_img),
                         tf.ones(power_img.shape) * 100,
                         power_img)
    min_snr = tf.math.reduce_min(snr_img)
    max_snr = tf.math.reduce_max(snr_img)
    min_power = tf.math.reduce_min(power_img)
    max_power = tf.math.reduce_max(power_img)
    value_img = (power_img - min_power) / (max_power - min_power)
    redgreen_img = (snr_img - min_snr) / (max_snr - min_snr)
    red_img = redgreen_img * value_img
    blue_img = (1 - redgreen_img) * value_img
    zero_img = tf.zeros(red_img.shape)
    input_img = tf.concat([red_img, zero_img, blue_img], axis=3)
    return input_img

  def _create_input_images(self):
    self.input_images = []
    self.y_trues = []
    self.input_descs = []
    for example in self.ex_img_batch:
      input_example, (target_phons_example, _) = example
      self.y_trues.append(self._img_from_y(target_phons_example))

      bin_descs = []
      for bin_idx, loudness in enumerate(target_phons_example[0].numpy()):
        if loudness > 0:
          bin_descs.append("%f Hz@%f phons" % (bin_idx * self.hz_per_bin,
                                               loudness))
      self.input_descs.append(", ".join(bin_descs))
      input_img = self._compute_rgb_image(input_example)
      self.input_images.append(input_img)

  def _img_from_y(self, y):
    max_pixel = tf.math.reduce_max(y)
    min_pixel = tf.math.reduce_min(y)
    y = (y - min_pixel) / (max_pixel - min_pixel)
    if len(y.get_shape().as_list()) == 1:
      y = tf.expand_dims(y, axis=0)
    padded_y = tf.tile(y, [self.thickness + 1, 1])
    img = tf.expand_dims(tf.expand_dims(padded_y, axis=2), axis=0)
    return img

  def _mask_img_from_y(self, y):
    # TODO(lauraruis): visualize mask predictions in tensorboard
    pass

  def _add_input_images(self):
    with self.file_writer.as_default():
      for idx, img in enumerate(self.input_images):
        tf.summary.image("Input %d" % idx,
                         img,
                         step=0,
                         description=self.input_descs[idx])
      for idx, truth in enumerate(self.y_trues):
        tf.summary.image("y_true %d" % idx,
                         truth,
                         step=0,
                         description=self.input_descs[idx])

  def _add_output_images(self):
    all_y_preds = []
    for example in self.ex_img_batch:
      input_example, (_, _) = example
      predicted_phons = self.model.predict(input_example)
      y = tf.squeeze(predicted_phons)
      all_y_preds.append(self._img_from_y(y))

    with self.file_writer.as_default():
      for idx, pred in enumerate(all_y_preds):
        tf.summary.image("y_pred %d" % idx,
                         pred,
                         step=self.model.step.numpy())


def binary_search(arr, item):
  """Finds closest index to the left of an item in arr."""
  low = 0
  high = len(arr) - 1
  mid = 0

  while low <= high:
    mid = (high + low) // 2
    # Check if item is present at mid
    if arr[mid] < item:
      low = mid
    # If item is greater, ignore left half
    elif arr[mid] > item:
      high = mid
    # If item is smaller, ignore right half
    else:
      return mid
    if arr[high] <= item:
      return high

    if arr[low] <= item < arr[low + 1]:
      return low
  return mid


def frequency_to_bin(frequency: float, sample_rate: int,
                     window_size: int) -> int:
  return round(frequency / (sample_rate / window_size))


def bin_to_frequency(frequency_bin: int, sample_rate: int,
                     window_size: int) -> float:
  return frequency_bin * (sample_rate / window_size)


def bin_to_cb(frequency_bin: int, critical_bands: List[int],
              sample_rate: int, window_size: int) -> int:
  assert frequency_bin >= 0, "Frequency bins must start from 1 instead of 0."
  frequency = bin_to_frequency(frequency_bin, sample_rate, window_size)
  cb = binary_search(critical_bands, frequency)
  return cb


def frequency_to_cb(frequency: float, critical_bands: List[int],
                    sample_rate: int, window_size: int) -> int:
  cb = binary_search(critical_bands, frequency)
  return cb

