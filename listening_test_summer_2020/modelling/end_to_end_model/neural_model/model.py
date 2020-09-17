# Lint as: python3
"""A TensorFlow model to predict loudness in phons of spectra of sounds."""
from typing import Tuple

from absl import logging

import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()


class LoudnessPredictor(tf.keras.Model):
  """Model to predict the loudness in phons of an audio spectrum."""

  def __init__(self, num_rows_channel_kernel: int, num_cols_channel_kernel: int,
               num_filters_channels: int, num_rows_bin_kernel: int,
               num_cols_bin_kernel: int, num_filters_bins: int,
               frequency_bins: int, carfac_channels: int, dropout_p: float,
               use_channels: bool, seed: int,
               name="LoudnessPredictor"):
    super(LoudnessPredictor, self).__init__(name=name)
    self.frequency_bins = frequency_bins
    self.carfac_channels = carfac_channels
    self.relu = tf.keras.activations.relu
    self.tanh = tf.keras.activations.tanh
    self.sigmoid = tf.keras.activations.sigmoid
    self.softmax = tf.nn.softmax
    self.num_filters_channels = num_filters_channels
    self.num_filters_bins = num_filters_bins
    self.use_channels = use_channels
    self.total_num_filters = num_filters_bins
    self.use_channels = use_channels
    if use_channels:
      self.total_num_filters += num_filters_channels
      channels = carfac_channels
      self.features_to_hidden = tf.keras.layers.Dense(
          20,
          input_shape=((channels + 1) * 2,),
          name="fc_features_to_mask",
          bias_initializer=tf.keras.initializers.Ones())
      self.bilstm_one = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50))
      self.bilstm_two = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50))
      self.hidden_to_logits = tf.keras.layers.Dense(
          1,
          input_shape=(200,),
          name="hidden_to_logits_one",
          bias_initializer=tf.keras.initializers.Ones())
    else:
      channels = 1
      self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100))
      # self.features_to_hidden = tf.keras.layers.Dense(
      #     100, input_shape=(25 + 1,), name="fc_features_to_mask")
      self.hidden_to_logits = tf.keras.layers.Dense(
          1, input_shape=(100,), name="hidden_to_logits_one")
    self.step = tf.Variable(0, trainable=False, name="step")
    self.dropout = tf.keras.layers.Dropout(dropout_p,
                                           seed=seed)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Forward pass of the model.

    Takes as input a sound processed by CARFAC, FFT, and a power calculation
    resulting in a tensor of shape [bsz, CARFAC channels, frequency bins, 2]
    where 2 is signal power and noise power in dB.

    Args:
      inputs: a batch of inputs [bsz, CARFAC channels, frequency bins, 2]

    Returns:
      A tuple of tensors containing the loundess predictions in phons per
      frequency bin and a soft mask over these bins.
    """
    (_, channels, bins, feature_dim) = inputs.get_shape().as_list()
    if not self.use_channels:
      # [bsz, 1, bins, dim] -> [bsz, 1, bins * dim]

      # [bsz, 1, bins, 1] -> [bsz, bins, 1]
      inputs = tf.squeeze(inputs, axis=1)

      # [bsz, bins, 1] -> [bsz, hidden_dim]
      hidden = self.bilstm(inputs)
      hidden = self.dropout(hidden)

      # [bsz, hidden_dim] -> [bsz, 1]
      output = self.relu(self.hidden_to_logits(hidden))

      # [bsz, 1] -> [bsz]
      output = tf.squeeze(output, axis=1)
    else:
      # [bsz, channels, bins, dim] -> [bsz, bins, channels, dim]
      inputs = tf.transpose(inputs, perm=[0, 2, 1, 3])

      # [bsz, bins, 1, dim]
      conditioning = inputs[:, :, -1, :]

      # [bsz, bins, 1, dim]
      coefs = inputs[:, :, -2, :]

      # [bsz, bins + 1]
      coefs = tf.concat([coefs,
                         tf.expand_dims(conditioning[:, 0], axis=2)],
                        axis=1)

      # [bsz, bins + 1, 1] -> [bsz, hidden_dim]
      hidden_two = self.bilstm_two(coefs)

      # [bsz, bins, carfac_channels, dim]
      inputs = inputs[:, :, :-2, :]
      inputs = self.dropout(inputs)

      # [bsz, bins, carfac_shannels, dim] -> [bsz, bins, carfac_channels * dim]
      inputs = tf.reshape(inputs, [-1, bins, (channels - 2) * feature_dim])

      # [bsz, bins, carfac_channels * dim] -> [bsz, bins, hidden_dim]
      hidden = self.features_to_hidden(inputs)

       # [bsz, bins, hidden_dim] -> [bsz, hidden_dim]
      hidden_one = self.bilstm_one(hidden)
      hidden = self.dropout(hidden)
      #       (_, bins, feature_dim) = hidden.get_shape().as_list()

      #       # [bsz, bins, hidden_dim] -> [bsz, bins *  hidden_dim]
      #       hidden_flat = tf.reshape(hidden, [-1, bins * feature_dim])

      # [bsz, hidden_dim] -> [bsz, hidden_dim + hidden_dim]
      hidden = tf.concat([hidden_one, hidden_two], axis=1)

      # [bsz, hidden_dim + hidden_dim] -> [bsz, 1]
      output = self.relu(self.hidden_to_logits(hidden))

    if len(output.get_shape().as_list()) == 1:
      output = tf.expand_dims(output, axis=0)

    return output
