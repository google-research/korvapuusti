# Lint as: python3
"""Train a loudness predictor with custom loss and metrics."""
from typing import Tuple

from absl import logging
import tensorflow.compat.v2 as tf

from model import LoudnessPredictor

tf.enable_v2_behavior()


def binary_cross_enty(y_true, y_pred):
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)
  cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 10, name=None)
  return cost


def multi_output_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.float32:
  """Calculates the MSE and Binary Cross Entropy over the outputs."""
  tf.print("Sum of actual masking: ", tf.reduce_sum(y_true))
  tf.print("Sum of predicted masking: ", tf.reduce_sum(y_pred))
  # loss_multiplier = tf.where(tf.greater(y_true, tf.constant(5.)), tf.constant(10.),
  #                        tf.constant(1.))
  loss = tf.keras.losses.mean_squared_error(y_true,
                                            y_pred)
  # tf.print("Y true: ", y_true)
  # tf.print("Loss multiplier: ", loss_multiplier)
  # loss *= tf.cast(loss_multiplier, dtype=tf.float32)
  return tf.reduce_mean(loss)


class MSEMetric(tf.keras.metrics.Metric):
  """Calculates the MSE for predictions."""

  def __init__(self, name='MSEMetric', **kwargs):
    super(MSEMetric, self).__init__(name=name, **kwargs)
    self.mse = self.add_weight(name='mse', initializer='zeros')
    self.step = self.add_weight(name='step', initializer='zeros')

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                   sample_weight=None):
    mse = tf.keras.metrics.mse(y_true, y_pred)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, 'float32')
      mse = tf.multiply(mse, sample_weight)
    self.mse.assign_add(tf.reduce_mean(mse))
    self.step.assign_add(tf.constant(1., dtype=tf.float32))

  def result(self) -> float:
    return self.mse / self.step

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.mse.assign(0.)
    self.step.assign(0)


class AccuracyMetric(tf.keras.metrics.Metric):
  """Calculates the accuracy for predictions."""

  def __init__(self, name='AccuracyMetric', **kwargs):
    super(AccuracyMetric, self).__init__(name=name, **kwargs)
    self.accuracy = self.add_weight(name='accuracy', initializer='zeros')
    self.step = self.add_weight(name='step', initializer='zeros')

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                   sample_weight=None):
    # predicted_mask = y_pred[:, :, 1]
    # actual_mask = y_true[:, :, 1]
    # predicted_mask = tf.squeeze(tf.keras.activations.sigmoid(predicted_mask))
    # actual_mask = tf.squeeze(actual_mask)
    # tf.print("In accuracy metric, sum of predicted mask and actual mask",
    #          tf.reduce_sum(predicted_mask), tf.reduce_sum(actual_mask))
    actual_mask = tf.cast(y_true, dtype=tf.float32)
    predicted_mask = tf.cast(tf.keras.activations.sigmoid(y_pred),
                             dtype=tf.float32)
    actual_mask = tf.squeeze(actual_mask)
    predicted_mask = tf.squeeze(predicted_mask)
    accuracy = tf.keras.metrics.binary_accuracy(actual_mask,
                                                predicted_mask,
                                                threshold=0.5)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, 'float32')
      accuracy = tf.multiply(accuracy, sample_weight)
    self.accuracy.assign_add(tf.reduce_mean(accuracy))
    self.step.assign_add(tf.constant(1., dtype=tf.float32))

  def result(self) -> float:
    return self.accuracy / self.step

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.accuracy.assign(0)
    self.step.assign(0)


class FalsePositives(tf.keras.metrics.Metric):
  """Calculates the accuracy for predictions."""

  def __init__(self, name='FalsePositives', **kwargs):
    super(FalsePositives, self).__init__(name=name, **kwargs)
    self.fp = self.add_weight(name='falsepositives', initializer='zeros')
    self.fp_metric = tf.keras.metrics.FalsePositives()
    self.step = self.add_weight(name='step', initializer='zeros')

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                   sample_weight=None):
    # predicted_mask = y_pred[:, :, 1]
    # actual_mask = y_true[:, :, 1]
    # predicted_mask = tf.squeeze(tf.keras.activations.sigmoid(predicted_mask))
    # actual_mask = tf.squeeze(actual_mask)
    # tf.print("In accuracy metric, sum of predicted mask and actual mask",
    #          tf.reduce_sum(predicted_mask), tf.reduce_sum(actual_mask))
    actual_mask = tf.cast(y_true, dtype=tf.float32)
    predicted_mask = tf.cast(tf.keras.activations.sigmoid(y_pred),
                             dtype=tf.float32)
    actual_mask = tf.squeeze(actual_mask)
    predicted_mask = tf.squeeze(predicted_mask)
    hard_mask = tf.where(tf.greater(predicted_mask, 0.5),
                         tf.constant(1, dtype=tf.float32),
                         tf.constant(0, dtype=tf.float32))
    hard_mask = tf.squeeze(hard_mask)
    self.fp_metric.update_state(actual_mask, hard_mask)
    fp = self.fp_metric.result()
    self.fp_metric.reset_states()
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, 'float32')
      fp = tf.multiply(fp, sample_weight)
    self.fp.assign_add(fp)
    self.step.assign_add(tf.constant(1., dtype=tf.float32))

  def result(self) -> float:
    return self.fp / self.step

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.fp.assign(0)
    self.step.assign(0)
    self.fp_metric.reset_states()


class FalseNegatives(tf.keras.metrics.Metric):
  """Calculates the accuracy for predictions."""

  def __init__(self, name='FalseNegatives', **kwargs):
    super(FalseNegatives, self).__init__(name=name, **kwargs)
    self.fn = self.add_weight(name='falsenegatives', initializer='zeros')
    self.fn_metric = tf.keras.metrics.FalseNegatives()
    self.step = self.add_weight(name='step', initializer='zeros')

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                   sample_weight=None):
    # predicted_mask = y_pred[:, :, 1]
    # actual_mask = y_true[:, :, 1]
    # predicted_mask = tf.squeeze(tf.keras.activations.sigmoid(predicted_mask))
    # actual_mask = tf.squeeze(actual_mask)
    # tf.print("In accuracy metric, sum of predicted mask and actual mask",
    #          tf.reduce_sum(predicted_mask), tf.reduce_sum(actual_mask))
    actual_mask = tf.cast(y_true, dtype=tf.float32)
    predicted_mask = tf.cast(tf.keras.activations.sigmoid(y_pred),
                             dtype=tf.float32)
    actual_mask = tf.squeeze(actual_mask)
    predicted_mask = tf.squeeze(predicted_mask)
    hard_mask = tf.where(tf.greater(predicted_mask, 0.5),
                         tf.constant(1, dtype=tf.float32),
                         tf.constant(0, dtype=tf.float32))
    hard_mask = tf.squeeze(hard_mask)
    self.fn_metric.update_state(actual_mask, hard_mask)
    fn = self.fn_metric.result()
    self.fn_metric.reset_states()
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, 'float32')
      fn = tf.multiply(fn, sample_weight)
    self.fn.assign_add(fn)
    self.step.assign_add(tf.constant(1., dtype=tf.float32))

  def result(self) -> float:
    return self.fn / self.step

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.fn.assign(0)
    self.step.assign(0)
    self.fn_metric.reset_states()


class TruePositives(tf.keras.metrics.Metric):
  """Calculates the accuracy for predictions."""

  def __init__(self, name='TruePositives', **kwargs):
    super(TruePositives, self).__init__(name=name, **kwargs)
    self.tp = self.add_weight(name='truepositives', initializer='zeros')
    self.tp_metric = tf.keras.metrics.TruePositives()
    self.step = self.add_weight(name='step', initializer='zeros')

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                   sample_weight=None):
    # predicted_mask = y_pred[:, :, 1]
    # actual_mask = y_true[:, :, 1]
    # predicted_mask = tf.squeeze(tf.keras.activations.sigmoid(predicted_mask))
    # actual_mask = tf.squeeze(actual_mask)
    # tf.print("In accuracy metric, sum of predicted mask and actual mask",
    #          tf.reduce_sum(predicted_mask), tf.reduce_sum(actual_mask))
    actual_mask = tf.cast(y_true, dtype=tf.float32)
    predicted_mask = tf.cast(tf.keras.activations.sigmoid(y_pred),
                             dtype=tf.float32)
    actual_mask = tf.squeeze(actual_mask)
    predicted_mask = tf.squeeze(predicted_mask)
    hard_mask = tf.where(tf.greater(predicted_mask, 0.5),
                         tf.constant(1, dtype=tf.float32),
                         tf.constant(0, dtype=tf.float32))
    hard_mask = tf.squeeze(hard_mask)
    self.tp_metric.update_state(actual_mask, hard_mask)
    tp = self.tp_metric.result()
    self.tp_metric.reset_states()
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, 'float32')
      tp = tf.multiply(tp, sample_weight)
    self.tp.assign_add(tp)
    self.step.assign_add(tf.constant(1., dtype=tf.float32))

  def result(self) -> float:
    return self.tp / self.step

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.tp.assign(0)
    self.step.assign(0)
    self.tp_metric.reset_states()


class TrueNegatives(tf.keras.metrics.Metric):
  """Calculates the accuracy for predictions."""

  def __init__(self, name='TrueNegatives', **kwargs):
    super(TrueNegatives, self).__init__(name=name, **kwargs)
    self.tn = self.add_weight(name='truenegatives', initializer='zeros')
    self.tn_metric = tf.keras.metrics.TrueNegatives()
    self.step = self.add_weight(name='step', initializer='zeros')

  def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                   sample_weight=None):
    # predicted_mask = y_pred[:, :, 1]
    # actual_mask = y_true[:, :, 1]
    # predicted_mask = tf.squeeze(tf.keras.activations.sigmoid(predicted_mask))
    # actual_mask = tf.squeeze(actual_mask)
    # tf.print("In accuracy metric, sum of predicted mask and actual mask",
    #          tf.reduce_sum(predicted_mask), tf.reduce_sum(actual_mask))
    actual_mask = tf.cast(y_true, dtype=tf.float32)
    predicted_mask = tf.cast(tf.keras.activations.sigmoid(y_pred),
                             dtype=tf.float32)
    actual_mask = tf.squeeze(actual_mask)
    predicted_mask = tf.squeeze(predicted_mask)
    hard_mask = tf.where(tf.greater(predicted_mask, 0.5),
                         tf.constant(1, dtype=tf.float32),
                         tf.constant(0, dtype=tf.float32))
    hard_mask = tf.squeeze(hard_mask)
    self.tn_metric.update_state(actual_mask, hard_mask)
    tn = self.tn_metric.result()
    self.tn_metric.reset_states()
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, 'float32')
      tn = tf.multiply(tn, sample_weight)
    self.tn.assign_add(tn)
    self.step.assign_add(tf.constant(1., dtype=tf.float32))

  def result(self) -> float:
    return self.tn / self.step

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.tn.assign(0)
    self.step.assign(0)
    self.tn_metric.reset_states()


def train(model: LoudnessPredictor, training_data: tf.data.Dataset,
          validation_data: tf.data.Dataset, learning_rate: float,
          epochs: int, steps_per_epoch: int, callbacks=None):
  """Train a loudness predictor on training data.

  Args:
    model: an instance of LoudnessPredictor
    training_data: data to train the model on.
    validation_data: data to validate the model with.
    learning_rate: the learning rate to use during training.
    epochs: the number of passes to do over the training data.
    steps_per_epoch: the number of iterations to do per epoch (n_examples / bsz)
    callbacks: a list of callbacks for TensorFlow
  """
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer,
                loss=multi_output_loss,
                metrics=[MSEMetric()])
  logging.info('Compiled model')
  model.fit(training_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch)
  return
