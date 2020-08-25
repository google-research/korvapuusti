# Lint as: python3
"""Tests for model."""
from typing import List

import tensorflow.compat.v2 as tf

import model


class ModelTest(tf.test.TestCase):

  def setUp(self):
    """Sets up inputs and model."""
    super(ModelTest, self).setUp()
    self.frequency_bins = 1024
    self.carfac_channels = 83
    self.test_input_1 = tf.random.uniform(shape=[3, self.carfac_channels,
                                                 25, 3],
                                          dtype=tf.float32)
    self.test_input_2 = tf.random.uniform(shape=[1, 1,
                                                 25 + 1, 1],
                                          dtype=tf.float32)
    self.model_carfac = model.LoudnessPredictor(
        frequency_bins=1024,
        carfac_channels=83,
        num_rows_channel_kernel=5,
        num_cols_channel_kernel=1,
        num_filters_channels=5,
        num_rows_bin_kernel=1,
        num_cols_bin_kernel=5,
        num_filters_bins=7,
        dropout_p=0.,
        use_channels=True,
        seed=0)
    self.model_nocarfac = model.LoudnessPredictor(
        frequency_bins=1024,
        carfac_channels=83,
        num_rows_channel_kernel=5,
        num_cols_channel_kernel=1,
        num_filters_channels=5,
        num_rows_bin_kernel=1,
        num_cols_bin_kernel=5,
        num_filters_bins=7,
        dropout_p=0.,
        use_channels=False,
        seed=0)

  def check_model_prediction(self, expected_dimensions: List[int],
                             example_input: tf.Tensor, model) -> {}:
    """Checks dimensions of the output of a forward pass through the model."""
    output_mask = model.predict(example_input)
    # output_features, output_mask = output[:, :, 0], output[:, :, 1]
    # output_features_shape = output_features.shape
    output_mask_shape = output_mask.shape
    # self.assertAllEqual(expected_dimensions, output_features_shape)
    self.assertAllEqual(expected_dimensions, output_mask_shape)
    self.assertAllInRange(tf.keras.activations.sigmoid(output_mask), 0, 1)

  def test_model_predictions(self) -> {}:
    """Checks the output dimensions of examples with batch size 3 and 1."""
    expected_output_shape_1 = [3, 1]
    expected_output_shape_2 = [1, 1]
    self.check_model_prediction(expected_output_shape_1, self.test_input_1,
                                self.model_carfac)
    self.check_model_prediction(expected_output_shape_2, self.test_input_2,
                                self.model_nocarfac)
    self.assertEqual(1, 2)


if __name__ == "__main__":
  tf.test.main()
