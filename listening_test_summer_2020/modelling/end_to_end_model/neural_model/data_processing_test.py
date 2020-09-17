# Lint as: python3
"""Tests for loudness_predictor."""
import tensorflow.compat.v2 as tf

import data_processing


class DataProcessingTest(tf.test.TestCase):

  def setUp(self):
    """Sets up inputs, predictions and targets."""
    super(DataProcessingTest, self).setUp()
    self.expected_batch_size = 3
    self.data = data_processing.get_datasets(
        "testing_data/testdataraw_0000000-0000020*",
        self.expected_batch_size, carfac=True)
    self.data_nocarfac = data_processing.get_datasets(
        "testing_data/testdataraw_0000000-0000020*",
        self.expected_batch_size, carfac=False)

  def check_example_dimensions(self, example: tf.Tensor,
                               expected_batch_size: int, channels: int) -> {}:
    expected_frequency_bins = 1
    input_example, target = example
    # target_example, target_mask = target[:, :, 0], target[:, :, 1]
    target_mask = target
    actual_channels = input_example.shape[1]
    actual_batch_size, actual_frequency_bins = target.shape
    self.assertEqual(actual_channels, channels)
    self.assertEqual(actual_batch_size, expected_batch_size)
    self.assertEqual(actual_frequency_bins, expected_frequency_bins)
    self.assertAllEqual(target_mask.shape[1], expected_frequency_bins)

  def test_get_frequency_mask(self):
    """Checks that target for frequency mask gets constructed correctly."""
    test_phons_per_frequency_bin = tf.constant([0, 12, 0, 0, 1],
                                               dtype=tf.float32)
    expected_frequency_mask = tf.constant([0, 1, 0, 0, 1],
                                          dtype=tf.float32)
    # expected_frequency_mask = tf.constant([[1., 0.],
    #                                        [0., 1.],
    #                                        [1., 0.],
    #                                        [1., 0.],
    #                                        [0., 1.]],
    #                                       dtype=tf.float32)
    actual_frequency_mask = data_processing._get_frequency_mask(
        test_phons_per_frequency_bin, 5)
    self.assertAllEqual(actual_frequency_mask, expected_frequency_mask)

  # def test_get_spl_tensor(self):
  #   """Checks that spls get scattered correctly over a tensor."""
  #   frequencies = tf.constant([4., 1.])
  #   spls = tf.constant([30., 50.])
  #   bins = 6
  #   expected_spl_tensor = tf.constant([0., 50., 0., 0., 30., 0.],
  #                                     dtype=tf.float32)
  #   actual_spl_tensor = data_processing._get_spl_tensor(bins, frequencies, spls)
  #   self.assertAllEqual(expected_spl_tensor, actual_spl_tensor)

  def test_get_loudness_tensor(self):
    """Checks that spls get scattered correctly over a tensor."""
    frequencies = [21., 15000.]
    spls = [20, 90]
    expected_loudness_tensor = tf.constant([1., 0., 0., 0.],
                                           dtype=tf.float32)
    bins = 4
    samplerate = 44100
    window_size = 9
    probe_frequency = 21.
    probe_loudness = 50.
    actual_loudness_tensor = data_processing._get_loudness_tensor(
        bins, frequencies, probe_frequency, probe_loudness, spls, samplerate,
        window_size)
    self.assertAllEqual(expected_loudness_tensor, actual_loudness_tensor)

  def test_get_datasets(self):
    """Tests whether shard of files with data get processed as expected.

    Checks dimensions of processed data files and whether validation set
    is always the same data.
    """
    expected_target_validation_example = tf.constant([59])
    for example in self.data["train"].take(1):
      self.check_example_dimensions(
          example, expected_batch_size=self.expected_batch_size,
          channels=86)
    for example in self.data_nocarfac["train"].take(1):
      self.check_example_dimensions(
          example, expected_batch_size=self.expected_batch_size,
          channels=1)
    for example in self.data["validate"].take(1):
      self.check_example_dimensions(example, expected_batch_size=1,
                                    channels=86)
    for example in self.data_nocarfac["validate"].take(1):
      self.check_example_dimensions(example, expected_batch_size=1,
                                    channels=1)
      # target_example = example[0]
      # nonzero_mask = tf.not_equal(target_example,
                                  # tf.constant(0, dtype=tf.float32))
      # self.assertAllEqual(target_example[nonzero_mask].numpy(),
                          # expected_target_validation_example)

if __name__ == "__main__":
  tf.test.main()
