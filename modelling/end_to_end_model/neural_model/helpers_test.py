# Lint as: python3
"""Tests for helpers."""

from absl.testing import absltest
import tensorflow.compat.v2 as tf

import helpers
import test_helpers


class HelpersTest(tf.test.TestCase):

  def setUp(self):
    """Sets up inputs, predictions and targets."""
    super(HelpersTest, self).setUp()
#     self.bins = 4
#     self.channels = 3
#     self.batch_size = 1
#     self._fake_batch = test_helpers.generate_fake_batch(
#         batch_size=self.batch_size, channels=self.channels, bins=self.bins)
#     self.image_writer_callback = helpers.ImageWriterCallback(
#         log_dir=absltest.get_default_test_tmpdir(),
#         channels=self.channels,
#         bins=self.bins,
#         ex_img_batch=self._fake_batch,
#         sample_rate=44100)

#   def test_compute_rgb_image(self):
#     output_image = self.image_writer_callback._compute_rgb_image(
#     self._fake_batch[0][0])
#     self.assertAllEqual([1, self.channels, self.bins, 3], output_image.shape)
#     self.assertAllEqual(
#         tf.zeros(shape=[1, self.channels, self.bins], dtype=tf.float32),
#         output_image[:, :, :, 1])
#     self.assertAllInRange(output_image, lower_bound=0., upper_bound=1.)

#   def test_img_from_y(self):
#     output_image = self.image_writer_callback._img_from_y(
#         tf.squeeze(self._fake_batch[0][1][0], axis=1))
#     self.assertAllEqual(
#         [1, self.image_writer_callback.thickness + 1, self.bins, 1],
#         output_image.shape)
#     self.assertAllInRange(output_image, lower_bound=0, upper_bound=1)

  def test_bin_to_cb(self):
    critical_bands = [
        20, 100, 200, 300, 400, 505, 630, 770, 915, 1080, 1265, 1475, 1720,
        1990, 2310, 2690, 3125, 3675, 4350, 5250, 6350, 7650, 9400, 11750,
        15250, 20000
    ]
    sample_rate = 44100
    window_size = 2024
    freq_bins = [1, 2, 1024, 1025, 1023, 6]
    expected_cbs = [0, 0, len(critical_bands) - 1, len(critical_bands) - 1,
                    len(critical_bands) - 1, 1]
    for freq_bin, expected_cb in zip(freq_bins, expected_cbs):
      actual_cb = helpers.bin_to_cb(freq_bin, critical_bands, sample_rate,
                                    window_size)
      self.assertEqual(expected_cb, actual_cb)

  def test_roundtrip_bin_to_frequency(self):
    sample_rate = 44100
    window_size = 2024
    for freq_bin in range(1, int(window_size / 2) + 1):
      frequency = helpers.bin_to_frequency(freq_bin, sample_rate, window_size)
      actual_bin = helpers.frequency_to_bin(frequency, sample_rate, window_size)
      self.assertEqual(freq_bin, actual_bin)
    return

#   def test_binary_search(self):
#     pass


if __name__ == "__main__":
  tf.test.main()
