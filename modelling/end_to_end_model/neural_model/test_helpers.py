# Lint as: python3
"""Helper functions for testing"""
from typing import Tuple, List

import tensorflow.compat.v2 as tf


def generate_fake_example(bins=1024,
                          channels=83
                         ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
  example_input = tf.random.uniform(shape=[channels, bins, 2],
                                    dtype=tf.float32)
  example_target_phons = tf.random.uniform(shape=[bins, 1],
                                           dtype=tf.float32)
  example_target_mask = tf.random.uniform(shape=[bins, 1],
                                          minval=0, maxval=1,
                                          dtype=tf.float32)
  return (example_input, (example_target_phons, example_target_mask))


def generate_fake_batch(
    batch_size: int, channels=83, bins=1024,
) -> List[Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]]:
  example_batch = []
  for _ in range(batch_size):
    input_example, (
        target_phons_example, target_mask_example) = generate_fake_example(
            channels=channels, bins=bins)
    input_example = tf.expand_dims(input_example, axis=0)
    target_phons_example = tf.expand_dims(target_phons_example, axis=0)
    target_mask_example = tf.expand_dims(target_mask_example, axis=0)
    example_batch.append((input_example,
                          (target_phons_example,
                           target_mask_example)))
  return example_batch
