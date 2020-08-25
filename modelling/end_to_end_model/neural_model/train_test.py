# Lint as: python3
"""Tests for train."""
import tensorflow.compat.v2 as tf

import model
import test_helpers
import train
import data_processing
from helpers import StepIncrementingCallback


class TrainTest(tf.test.TestCase):

  def setUp(self):
    """Sets up inputs, predictions and targets."""
    super(TrainTest, self).setUp()
    self.mse_metric = train.MSEMetric()
    self.accuracy_metric = train.AccuracyMetric()
    self.model = model.LoudnessPredictor(
        frequency_bins=1024,
        carfac_channels=83,
        num_rows_channel_kernel=5,
        num_cols_channel_kernel=1,
        num_filters_channels=5,
        num_rows_bin_kernel=1,
        num_cols_bin_kernel=5,
        num_filters_bins=7,
        dropout_p=0.1,
        use_channels=True,
        seed=0)
    self.model_nocarfac = model.LoudnessPredictor(
        frequency_bins=1024,
        carfac_channels=1,
        num_rows_channel_kernel=5,
        num_cols_channel_kernel=1,
        num_filters_channels=5,
        num_rows_bin_kernel=1,
        num_cols_bin_kernel=5,
        num_filters_bins=7,
        dropout_p=0.1,
        use_channels=False,
        seed=0)

  def test_multi_output_loss(self):
    """Tests whether custom loss returns expected value."""
    expected_loss = 100. / 6 + 0.5665188 + 0.75
    test_phons_pred = tf.constant([[[50.], [0.], [40.]],
                                   [[0.], [0.], [0.]]], dtype=tf.float32)
    test_mask_pred = tf.constant([[[1.], [0.], [1.]],
                                  [[0.], [0.], [0.]]], dtype=tf.float32)
    test_phons_true = tf.constant([[[40.], [0.], [40.]],
                                   [[0.], [0.], [0.]]], dtype=tf.float32)
    test_mask_true = tf.constant([[[1.], [0.], [1.]],
                                  [[0.], [0.], [0.]]], dtype=tf.float32)
    test_mask_pred = tf.constant([[-5., 4.],
                                  [0., 100.]], dtype=tf.float32)
    test_mask_true = tf.constant([[1., 0.],
                                  [0., 1.]], dtype=tf.float32)
    test_mask_pred = tf.constant([0.8, 0.8, 0., 0.1], dtype=tf.float32)
    test_mask_true = tf.constant([1., 0., 1., 0.], dtype=tf.float32)
    expected_loss = 0.75 - 0.3275
    # y_true = tf.keras.layers.concatenate([
    #     tf.expand_dims(test_phons_true, axis=2),
    #     tf.expand_dims(test_mask_true, axis=2)
    # ],
    #                                      axis=2)
    # y_pred = tf.keras.layers.concatenate([
    #     tf.expand_dims(test_phons_pred, axis=2),
    #     tf.expand_dims(test_mask_pred, axis=2)
    # ],
    #                                      axis=2)
    actual_loss = train.multi_output_loss(test_mask_true, test_mask_pred)
    print("actual loss ", actual_loss)
    self.assertBetween(tf.math.abs(actual_loss - expected_loss), 0, 1e-2)

#   def test_multi_output_loss_mask(self):
#     """Tests whether custom loss returns expected value."""
#     expected_loss = 18.70100 - 1.3776989
#     test_phons_pred = tf.constant([[[50.], [0.], [40.]],
#                                    [[0.], [0.], [0.]]], dtype=tf.float32)
#     test_mask_pred = tf.constant([[[.5], [0.], [0.]],
#                                   [[0.], [0.], [0.]]], dtype=tf.float32)
#     test_phons_true = tf.constant([[[40.], [0.], [40.]],
#                                    [[0.], [0.], [0.]]], dtype=tf.float32)
#     test_mask_true = tf.constant([[[1.], [0.], [1.]],
#                                   [[0.], [0.], [0.]]], dtype=tf.float32)
#     y_true = tf.keras.layers.concatenate([
#         tf.expand_dims(test_phons_true, axis=2),
#         tf.expand_dims(test_mask_true, axis=2)
#     ],
#                                          axis=2)
#     y_pred = tf.keras.layers.concatenate([
#         tf.expand_dims(test_phons_pred, axis=2),
#         tf.expand_dims(test_mask_pred, axis=2)
#     ],
#                                          axis=2)
#     actual_loss = train.multi_output_loss(y_true, y_pred)
#     self.assertBetween(tf.math.abs(actual_loss - expected_loss), 0, 1e-3)

#   def test_multi_output_metrics(self):
#     """Tests whether custom metrics return expected values."""
#     expected_accuracy = 5. / 6
#     expected_mse = (100. + 25) / 6
#     test_phons_pred = tf.constant([[[50.], [0.], [40.]],
#                                    [[0.], [0.], [-5.]]], dtype=tf.float32)
#     test_mask_pred = tf.constant([[[.5], [.6], [1.]],
#                                   [[0.], [0.2], [0.49]]], dtype=tf.float32)
#     test_phons_true = tf.constant([[[40.], [0.], [40.]],
#                                    [[0.], [0.], [0.]]], dtype=tf.float32)
#     test_mask_true = tf.constant([[[1.], [0.], [1.]],
#                                   [[0.], [0.], [0.]]], dtype=tf.float32)
#     y_true = tf.keras.layers.concatenate([
#         tf.expand_dims(test_phons_true, axis=2),
#         tf.expand_dims(test_mask_true, axis=2)
#     ],
#                                          axis=2)
#     y_pred = tf.keras.layers.concatenate([
#         tf.expand_dims(test_phons_pred, axis=2),
#         tf.expand_dims(test_mask_pred, axis=2)
#     ],
#                                          axis=2)
#     self.mse_metric.reset_states()
#     self.accuracy_metric.reset_states()
#     self.mse_metric.update_state(y_true, y_pred)
#     self.accuracy_metric.update_state(y_true, y_pred)
#     actual_mse = self.mse_metric.result()
#     actual_accuracy = self.accuracy_metric.result()
#     self.assertBetween(tf.math.abs(actual_mse - expected_mse), 0, 1e-3)
#     self.assertBetween(tf.math.abs(actual_accuracy - expected_accuracy), 0, 1e-3)
#     self.assertEqual(self.mse_metric.step, 1)
#     self.assertEqual(self.accuracy_metric.step, 1)
#     self.mse_metric.reset_states()
#     self.accuracy_metric.reset_states()
#     self.assertEqual(self.mse_metric.step, 0)
#     self.assertEqual(self.accuracy_metric.step, 0)
#     self.assertAllEqual(self.mse_metric.mse, tf.zeros_like(self.mse_metric.mse))
#     self.assertAllEqual(self.accuracy_metric.accuracy,
#                         tf.zeros_like(self.accuracy_metric.accuracy))

  def test_accuracy_metric(self):
    # test_phons_pred_one = tf.constant(
    #     [[[50.], [0.], [40.]], [[0.], [0.], [-5.]]], dtype=tf.float32)
    # test_mask_pred_one = tf.constant(
    #     [[[.5], [.6], [1.]], [[0.], [0.2], [0.49]]], dtype=tf.float32)
    # test_phons_true_one = tf.constant(
    #     [[[40.], [0.], [40.]], [[0.], [0.], [0.]]], dtype=tf.float32)
    # test_mask_true_one = tf.constant([[[1.], [0.], [1.]], [[0.], [0.], [0.]]],
    #                                  dtype=tf.float32)
    # y_true_one = tf.keras.layers.concatenate([
    #     tf.expand_dims(test_phons_true_one, axis=2),
    #     tf.expand_dims(test_mask_true_one, axis=2)
    # ],
    #                                          axis=2)
    # y_pred_one = tf.keras.layers.concatenate([
    #     tf.expand_dims(test_phons_pred_one, axis=2),
    #     tf.expand_dims(test_mask_pred_one, axis=2)
    # ],
    #                                          axis=2)
    # expected_accuracy = ((5 / 6) + (2 / 6)) / 2
    # test_phons_pred_two = tf.constant(
    #     [[[50.], [0.], [40.]], [[0.], [0.], [-5.]]], dtype=tf.float32)
    # test_mask_pred_two = tf.constant(
    #     [[[.4], [.6], [.1]], [[0.8], [0.2], [0.49]]], dtype=tf.float32)
    # test_phons_true_two = tf.constant(
    #     [[[40.], [0.], [40.]], [[0.], [0.], [0.]]], dtype=tf.float32)
    # test_mask_true_two = tf.constant([[[1.], [0.], [1.]], [[0.], [0.], [0.]]],
    #                                  dtype=tf.float32)
    # y_true_two = tf.keras.layers.concatenate([
    #     tf.expand_dims(test_phons_true_two, axis=2),
    #     tf.expand_dims(test_mask_true_two, axis=2)
    # ],
    #                                          axis=2)
    # y_pred_two = tf.keras.layers.concatenate([
    #     tf.expand_dims(test_phons_pred_two, axis=2),
    #     tf.expand_dims(test_mask_pred_two, axis=2)
    # ],
    #                                          axis=2)
    test_mask_pred_one = tf.constant([[-5., 4.],
                                  [0., 100.]], dtype=tf.float32)
    test_mask_true_one = tf.constant([[1., 0.],
                                  [0., 1.]], dtype=tf.float32)
    test_mask_pred_one = tf.constant([100, -50, -100., -1000], dtype=tf.float32)
    test_mask_true_one = tf.constant([1., 0., 1., 0.], dtype=tf.float32)
    test_mask_pred_two = tf.constant([[-5., 4.],
                                  [102., 100.]], dtype=tf.float32)
    test_mask_true_two = tf.constant([[1., 0.],
                                  [0., 1.]], dtype=tf.float32)
    test_mask_pred_two = tf.constant([1000, 1000, -1000., 1000], dtype=tf.float32)
    test_mask_true_two = tf.constant([1., 0., 1., 0.], dtype=tf.float32)
    expected_accuracy = 0.5
    self.accuracy_metric.reset_states()
    self.accuracy_metric.update_state(test_mask_true_one, test_mask_pred_one)
    self.accuracy_metric.update_state(test_mask_true_two, test_mask_pred_two)
    actual_accuracy = self.accuracy_metric.result()
    self.assertBetween(tf.math.abs(actual_accuracy - expected_accuracy), 0, 1e-3)
    self.assertEqual(self.accuracy_metric.step, 2)
    self.accuracy_metric.reset_states()
    self.assertEqual(self.accuracy_metric.step, 0)
    self.assertAllEqual(self.accuracy_metric.accuracy,
                        tf.zeros_like(self.accuracy_metric.accuracy))

  def test_train_carfac(self):
    """Tests whether forward and backward pass run without errors."""
    data = data_processing.get_datasets(
        "testing_data/testdataraw_*",
        1, carfac=True)
    self.assertEqual(self.model.step.numpy(), 0)
    train.train(
        self.model,
        data["train"],
        data["validate"],
        learning_rate=1e-4,
        epochs=3,
        callbacks=[StepIncrementingCallback()],
        steps_per_epoch=3)
    self.assertEqual(self.model.step.numpy(), 9)
    self.assertEqual(1, 2)

  def test_train_nocarfac(self):
    """Tests whether forward and backward pass run without errors."""
    data = data_processing.get_datasets(
        "testing_data/testdataraw_*",
        1, carfac=False)
    self.assertEqual(self.model_nocarfac.step.numpy(), 0)
    train.train(
        self.model_nocarfac,
        data["train"],
        data["validate"],
        learning_rate=1e-4,
        epochs=3,
        callbacks=[StepIncrementingCallback()],
        steps_per_epoch=3)
    self.assertEqual(self.model_nocarfac.step.numpy(), 9)


if __name__ == "__main__":
  tf.test.main()
