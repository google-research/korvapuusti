# Lint as: python3
"""Functions to parse loundess data for computational processing with TF."""
from typing import Dict, Union, Tuple, List, Any
import json
import math
import glob
import numpy as np

from absl import logging
import tensorflow.compat.v2 as tf

import helpers

tf.enable_v2_behavior()


def _proto_feature_description():
  return {
      'FreqBin': tf.io.FixedLenFeature([], tf.int64),
      'Phons': tf.io.FixedLenFeature([], tf.float32),
      'SPL': tf.io.FixedLenFeature([], tf.float32),
      'Freq': tf.io.FixedLenFeature([], tf.float32),
      'Bins': tf.io.FixedLenFeature([], tf.int64),
      'Channels': tf.io.FixedLenFeature([], tf.int64),
      'SpectrumSignal': tf.io.VarLenFeature(tf.float32),
      'SpectrumNoise': tf.io.VarLenFeature(tf.float32),
      'SampleRate': tf.io.FixedLenFeature([], tf.int64),
  }


def _proto_feature_description_v2():
  return {
      'SPLs': tf.io.VarLenFeature(tf.float32),
      'Frequencies': tf.io.VarLenFeature(tf.float32),
      'FrequencyBins': tf.io.VarLenFeature(tf.int64),
      'Bins': tf.io.FixedLenFeature([], tf.int64),
      'Channels': tf.io.FixedLenFeature([], tf.int64),
      'SpectrumSignal': tf.io.VarLenFeature(tf.float32),
      'SpectrumNoise': tf.io.VarLenFeature(tf.float32),
      'SampleRate': tf.io.FixedLenFeature([], tf.int64),
  }


def _proto_feature_description_types():
  return {
      'samplerate': tf.int64,
      'frequencies': tf.float32,
      'spls': tf.float32,
      'targetloudness': tf.float32,
      'probefrequency': tf.float32,
      'probelevel': tf.int64,
      'probeloudness': tf.int64,
      'probemasking': tf.float32,
      'probecb': tf.float32,
      'channels': tf.int64,
      'bins': tf.int64,
      'spectrumsignalcarfac': tf.float32,
      'spectrumnoisecarfac': tf.float32,
      'summarizedsignal': tf.float32,
      'summarizednoise': tf.float32,
      'summarizedcoefficients': tf.float32,
      'spectrumsignal': tf.float32,
      'spectrumnoise': tf.float32,
      'buffer': tf.float32,
      'coefficients': tf.float32,
  }


def _proto_feature_description_shapes():
  return {
      'samplerate': (1,),
      'frequencies': (2,),
      'spls': (2,),
      'targetloudness': (1024,),
      'probefrequency': (1,),
      'probelevel': (1,),
      'probecb': (1,),
      'probeloudness': (1,),
      'probemasking': (1,),
      'channels': (1,),
      'bins': (1,),
      'spectrumsignalcarfac': (86016,),
      'spectrumnoisecarfac': (86016,),
      'summarizedsignal': (84,25),
      'summarizednoise': (84,25),
      'summarizedcoefficients': (25,),
      'spectrumsignal': (1024,),
      'spectrumnoise': (1024,),
      'buffer': (2048,),
      'coefficients': (1024,),
  }


def summarize_carfac_over_channels(example, critical_bands, channels, bins,
                                   sample_rate):
  signal = np.array(example["spectrumsignalcarfac"])
  noise = np.array(example["spectrumnoisecarfac"])
  coefficients = np.array(example["coefficients"])
  signal = signal.reshape((channels, bins))
  noise = noise.reshape((channels, bins))
  summarized_signal = np.zeros([channels, 25])
  summarized_noise = np.zeros([channels, 25])
  summarized_coefficients = np.zeros([1, 25])
  for k_bin in range(1, bins + 1):
    cb_bin = helpers.bin_to_cb(k_bin, critical_bands, sample_rate, bins * 2)
    summarized_signal[:, cb_bin - 1] += signal[:, k_bin - 1]
    summarized_noise[:, cb_bin - 1] += noise[:, k_bin - 1]
    summarized_coefficients[0, cb_bin - 1] += coefficients[k_bin - 1]
  return summarized_signal, summarized_noise, summarized_coefficients


def _parse_single_example(s):
  critical_bands = [
      20, 100, 200, 300, 400, 505, 630, 770, 915, 1080, 1265, 1475, 1720, 1990,
      2310, 2690, 3125, 3675, 4350, 5250, 6350, 7650, 9400, 11750, 15250, 20000
  ]
  if not isinstance(s['samplerate'], int):
    example_target_loudness = _get_loudness_tensor(
        int(s['bins'].numpy()[0]), list(s['frequencies'].numpy()),
        float(s['probefrequency'].numpy()[0]),
        int(s['probeloudness'].numpy()[0]),
        [int(spl) for spl in list(s['spls'].numpy())],
        int(s['samplerate'].numpy()),
        int(s['bins'].numpy()) * 2)
    probe_cb = helpers.frequency_to_cb(
        float(s['probefrequency'].numpy()[0]), critical_bands,
        int(s['samplerate'].numpy()),
        int(s['bins'].numpy()) * 2)
    s['summarizedsignal'], s['summarizednoise'], s['summarizedcoefficients'] = summarize_carfac_over_channels(
      s, critical_bands, 84, int(s['bins'].numpy()), int(s['samplerate'].numpy()))
  else:
    example_target_loudness = _get_loudness_tensor(s['bins'], s['frequencies'],
                                                   s['probefrequency'],
                                                   int(s['probeloudness']),
                                                   s['spls'], s['samplerate'],
                                                   s['bins'] * 2)
    probe_cb = helpers.frequency_to_cb(
        float(s['probefrequency']), critical_bands, s['samplerate'],
        s['bins'] * 2)
    s['summarizedsignal'], s['summarizednoise'], s['summarizedcoefficients'] = summarize_carfac_over_channels(
      s, critical_bands, 84, s['bins'], s['samplerate'])
  s['targetloudness'] = example_target_loudness
  s['probecb'] = probe_cb
  for key, value in s.items():
    if not isinstance(value, list):
      value = [value]
    s[key] = tf.reshape(
        tf.convert_to_tensor(
            value, dtype=_proto_feature_description_types()[key]),
        shape=_proto_feature_description_shapes()[key])
  return s


def _get_frequency_mask(phons_per_frequency_bin: tf.Tensor,
                        bins: int) -> tf.Tensor:
  # [bins, 2]
  # zero_label = tf.tile(tf.constant([[1., 0.]], dtype=tf.float32),
  #                      tf.constant([bins, 1], tf.int32))
  # # [bins, 2]
  # one_label = tf.tile(tf.constant([[0., 1.]], dtype=tf.float32),
  #                     tf.constant([bins, 1], tf.int32))
  # # [bins, 1]
  # labels = tf.expand_dims(tf.math.greater(phons_per_frequency_bin,
  #                          tf.constant([0], dtype=tf.float32)), axis=1)
  # # [bins, 2]
  # tiled_labels = tf.tile(labels, tf.constant([1, 2], tf.int32))
  # mask = tf.where(tiled_labels, one_label, zero_label)
  mask = tf.cast(tf.not_equal(phons_per_frequency_bin,
                              tf.constant([0.], dtype=tf.float32)),
                 dtype=tf.float32)
  # return tf.convert_to_tensor(mask, dtype=tf.float32)
  return mask


def _get_spl_tensor(bins: int, frequencies: tf.Tensor,
                    spls: tf.Tensor) -> tf.Tensor:
  indices = tf.expand_dims(tf.cast(frequencies, dtype=tf.int32), axis=1)
  return tf.scatter_nd(indices, updates=spls,
                       shape=tf.constant([bins]))


def _get_loudness_tensor(bins: int, frequencies: List[float],
                         probe_frequency: float,
                         probe_loudness: int,
                         spls: List[int], sample_rate: int,
                         window_size: int) -> tf.Tensor:
  target_tensor = [0 for i in range(bins)]
  for frequency, spl in zip(frequencies, spls):
    frequency = float(frequency)
    sample_rate = int(sample_rate)
    window_size = int(window_size)
    frequency_bin = helpers.frequency_to_bin(frequency,
                                             sample_rate, window_size)
    if frequency == probe_frequency:
      if probe_loudness > spl:
        probe_loudness = spl
      target_tensor[int(frequency_bin)] = 1
  return tf.convert_to_tensor(target_tensor, dtype=tf.float32)


def _split_single_example(
    example: Dict[str, tf.Tensor], bins: int,
    channels: int, carfac: bool) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
  bins = 25
  if carfac:
    example_input = tf.concat([
        tf.reshape(example['summarizedsignal'], [channels, bins, 1]),
        tf.reshape(example['summarizednoise'], [channels, bins, 1])
    ], axis=2)
    example_input = tf.reduce_sum(example_input, axis=2)
    # example_input = example_input  # Make all features > 0
    example_input = tf.concat([
        tf.expand_dims(example_input, axis=2),
        tf.reshape(example['summarizedcoefficients'], [1, bins, 1])
    ], axis=0)
  else:
    channels = 1
    example_input = tf.reshape(example['summarizedcoefficients'], [1, bins, 1])

  full_target = tf.clip_by_value(
      example['probemasking'], 0,
      tf.cast(example['probelevel'], dtype=tf.float32))
  if carfac:
    input_conditioning = tf.tile(
        tf.expand_dims(example['probecb'], axis=0),
        tf.constant([bins, 1], tf.int32))
    example_input = tf.concat([
        example_input,
        tf.expand_dims(input_conditioning, axis=0)
    ], axis=0)
  else:
    example_input = tf.concat([
        example_input,
        tf.expand_dims(tf.expand_dims(example['probecb'], axis=0), axis=2)
    ], axis=1)
  return (example_input, full_target)


def _split_single_example_test(
    example: Dict[str, tf.Tensor], bins: int,
    channels: int, carfac: bool) -> Tuple[Any, Tuple[Any, Any], Dict[str, Any]]:
  bins = 25
  if carfac:
    example_input = tf.concat([
        tf.reshape(example['summarizedsignal'], [channels, bins, 1]),
        tf.reshape(example['summarizednoise'], [channels, bins, 1])
    ], axis=2)
    example_input = tf.reduce_sum(example_input, axis=2)
    # example_input = example_input  # Make all features > 0
    example_input = tf.concat([
        tf.expand_dims(example_input, axis=2),
        tf.reshape(example['summarizedcoefficients'], [1, bins, 1])
    ], axis=0)
  else:
    channels = 1
    example_input = tf.reshape(example['summarizedcoefficients'], [1, bins, 1])

  full_target = tf.clip_by_value(
      example['probemasking'], 0,
      tf.cast(example['probelevel'], dtype=tf.float32))
  if carfac:
    input_conditioning = tf.tile(
        tf.expand_dims(example['probecb'], axis=0),
        tf.constant([bins, 1], tf.int32))
    example_input = tf.concat([
        example_input,
        tf.expand_dims(input_conditioning, axis=0)
    ], axis=0)
  else:
    example_input = tf.concat([
        example_input,
        tf.expand_dims(tf.expand_dims(example['probecb'], axis=0), axis=2)
    ], axis=1)
  rest_example = example
  return (example_input, full_target, rest_example)


def get_testdata(file_pattern: str, carfac: bool) -> Dict[str, Union[tf.data.Dataset, int]]:
  files = glob.glob(file_pattern)
  logging.info('Testing with %d files from %s', len(files), file_pattern)
  all_data_test = []
  for i, file in enumerate(files):
    with open(file, 'r') as infile:
      data = json.load(infile)
      all_data_test.extend(data)
  def meta_dict_gen_test():
    for ex in all_data_test:
      yield _parse_single_example(ex)

  test_ds = tf.data.Dataset.from_generator(
      meta_dict_gen_test,
      output_types=_proto_feature_description_types(),
      output_shapes=_proto_feature_description_shapes())

  bins = None
  channels = None
  sample_rate = None
  for e in test_ds.take(1):
    bins = e['bins'].numpy()
    channels = e['channels'].numpy()
    sample_rate = e['samplerate'].numpy()

  test_ds = test_ds.map(
      lambda ex: _split_single_example_test(ex, bins, channels, carfac)).batch(1)

  return {
      'test': test_ds,
      'bins': bins,
      'channels': channels,
      'sample_rate': sample_rate
  }


def get_datasets(file_pattern: str,
                 batch_size: int,
                 carfac: bool,
                 extra_file_pattern=None) -> Dict[str, Union[tf.data.Dataset, int]]:
  """Takes a shard of files and processes into a batched TF dataset.

  Each 50th file will be put into the validation set with batch size 1,
  the rest of the data will be the training set with the specified batch_size.

  Args:
    file_pattern: location of the data files.
    batch_size: number of examples to put in each batch in train set.

  Returns:
    The training set, validation set, the number of frequency bins, carfac
    channels, and the sample rate.
  """
  files = glob.glob(file_pattern)
  if extra_file_pattern:
    extra_files = glob.glob(extra_file_pattern)
    logging.info('Using %d extra files from %s',
                 len(extra_files), extra_file_pattern)
    files = extra_files + files
  logging.info('Training with %d files from %s', len(files), file_pattern)
  all_data_train = []
  all_data_validate = []
  for i, file in enumerate(files):
    with open(file, 'r') as infile:
      data = json.load(infile)
      if i == len(files) - 1:
        all_data_validate.extend(data)
      else:
        all_data_train.extend(data)

  def meta_dict_gen_train():
    for ex in all_data_train:
      yield _parse_single_example(ex)

  def meta_dict_gen_val():
    for ex in all_data_validate:
      yield _parse_single_example(ex)

  train_ds = tf.data.Dataset.from_generator(
      meta_dict_gen_train,
      output_types=_proto_feature_description_types(),
      output_shapes=_proto_feature_description_shapes())
  validate_ds = tf.data.Dataset.from_generator(
      meta_dict_gen_val,
      output_types=_proto_feature_description_types(),
      output_shapes=_proto_feature_description_shapes())

  bins = None
  channels = None
  sample_rate = None
  for e in validate_ds.take(1):
    bins = e['bins'].numpy()
    channels = e['channels'].numpy()
    sample_rate = e['samplerate'].numpy()

  validate_ds = validate_ds.map(
      lambda ex: _split_single_example(ex, bins, channels, carfac))
  validate_ds = validate_ds.batch(1)

  train_ds = train_ds.shuffle(batch_size)
  train_ds = train_ds.map(
      lambda ex: _split_single_example(ex, bins, channels, carfac))
  train_ds = train_ds.batch(batch_size, drop_remainder=True)
  train_ds = train_ds.repeat()


  return {
      'train': train_ds,
      'validate': validate_ds,
      'bins': bins,
      'channels': channels,
      'sample_rate': sample_rate
  }
