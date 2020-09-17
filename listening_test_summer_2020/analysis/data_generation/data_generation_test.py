# Lint as: python3
"""Tests for data_generation.

Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""
import collections
import itertools

import numpy as np

import data_generation
import unittest
import loudness


class DataGenerationTest(unittest.TestCase):

  def test_binary_search_even(self):
    """Tests for even arr."""
    arr = [20, 100, 200, 505]
    items = [80, 20, 100, 101, 250, 700, 505]
    expected_indices = [0, 0, 1, 1, 2, 3, 3]
    for item, expected_index in zip(items, expected_indices):
      actual_index = data_generation.binary_search(arr, item)
      self.assertEqual(expected_index, actual_index)

  def test_binary_search_uneven(self):
    """Tests for uneven arr."""
    arr = [20, 100, 200, 505, 1000]
    items = [80, 20, 100, 101, 250, 700, 505, 1000, 5000]
    expected_indices = [0, 0, 1, 1, 2, 3, 3, 4, 4]
    for item, expected_index in zip(items, expected_indices):
      actual_index = data_generation.binary_search(arr, item)
      self.assertEqual(expected_index, actual_index)

  def test_calculate_beat_range_low(self):
    """Test if beatrange gets calculated correctly at the low frequency."""
    frequency = data_generation.FREQ_A
    expected_beat_range = data_generation.BEAT_RANGE_A
    actual_beat_range = data_generation.calculate_beat_range(frequency)
    self.assertEqual(expected_beat_range, actual_beat_range)

  def test_calculate_beat_range_high(self):
    """Test if beatrange gets calculated correctly at the high frequency."""
    frequency = data_generation.FREQ_B
    expected_beat_range = data_generation.BEAT_RANGE_B
    actual_beat_range = data_generation.calculate_beat_range(frequency)
    self.assertEqual(expected_beat_range, actual_beat_range)

  def test_check_frequencies_fails_low(self):
    """Supposed to fail because a frequency is too low."""
    min_frequency, max_frequency = 20, 20000
    test_frequencies = np.array([10, 200, 5000, 20])
    check_bool = data_generation.check_frequencies(
        frequencies=test_frequencies,
        min_frequency=min_frequency,
        max_frequency=max_frequency)
    self.assertFalse(check_bool)

  def test_check_frequencies_fails_beatrange(self):
    """Supposed to fail because two frequencies are too close together."""
    min_frequency, max_frequency = 20, 20000
    test_frequencies = np.array([25, 200, 5000, 30])
    check_bool = data_generation.check_frequencies(
        frequencies=test_frequencies,
        min_frequency=min_frequency,
        max_frequency=max_frequency)
    self.assertFalse(check_bool)

  def test_check_frequencies_pass(self):
    """Supposed to pass."""
    min_frequency, max_frequency = 20, 20000
    test_frequencies = np.array([20000, 200, 5000, 20])
    check_bool = data_generation.check_frequencies(
        frequencies=test_frequencies,
        min_frequency=min_frequency,
        max_frequency=max_frequency)
    self.assertTrue(check_bool)

  def test_check_phons_fails_high(self):
    """Supposed to fail because a phons level is too high."""
    min_phons, max_phons = 0, 80
    phons = np.array([0, 81, 5])
    check_bool = data_generation.check_phons(phons, min_phons=min_phons,
                                             max_phons=max_phons)
    self.assertFalse(check_bool)

  def test_check_phons_fails_low(self):
    """Supposed to fail because a phons level is too low."""
    min_phons, max_phons = 0, 80
    phons = np.array([-1, 79, 5])
    check_bool = data_generation.check_phons(phons, min_phons=min_phons,
                                             max_phons=max_phons)
    self.assertFalse(check_bool)

  def test_check_phons_pass(self):
    """Supposed to pass."""
    min_phons, max_phons = 0, 80
    phons = np.array([0, 30, 80])
    check_bool = data_generation.check_phons(phons, min_phons=min_phons,
                                             max_phons=max_phons)
    self.assertTrue(check_bool)

  def test_generate_iso_repro_examples(self):
    """Tests that the ISO examples are on the equal loudness curves."""
    num_examples_per_loudness_curve = 3
    num_curves = 2
    clip_db = 80
    allowed_phons_diff = 1
    data = data_generation.generate_iso_repro_examples(
        num_examples_per_curve=num_examples_per_loudness_curve,
        num_curves=num_curves,
        clip_db=clip_db)
    expected_phons = set([50, 60])
    self.assertEqual(len(data), num_curves)
    num_examples = 0
    for phons_level, examples in data.items():
      self.assertIn(phons_level, expected_phons)
      level = examples["ref1000_spl"]
      self.assertLessEqual(level, clip_db)
      actual_phons_ref = np.round(loudness.spl_to_loudness(level, 1000))
      phons_diff = abs(actual_phons_ref - phons_level)
      self.assertLessEqual(phons_diff, allowed_phons_diff)
      for other_tone in examples["other_tones"]:
        actual_phons = np.round(
            loudness.spl_to_loudness(other_tone["level"],
                                     other_tone["frequency"]))
        phons_diff = abs(actual_phons - phons_level)
        self.assertLessEqual(phons_diff, allowed_phons_diff)
        self.assertLessEqual(other_tone["level"], clip_db)
        num_examples += 1
    self.assertEqual(num_examples, num_curves * num_examples_per_loudness_curve)

  def test_sample_n_per_cb(self):
    """Test if the correct number of tones per critical band gets generated."""
    critical_bands = [15, 20, 50, 100]
    examples_per_cb = 200
    generated_tones_per_cb = data_generation.sample_n_per_cb(examples_per_cb,
                                                             critical_bands)
    tone_counter = collections.Counter()
    for cb, tone in generated_tones_per_cb:
      self.assertLessEqual(critical_bands[cb], tone)
      self.assertGreater(critical_bands[cb + 1], tone)
      tone_counter[cb] += 1
    for count in tone_counter.values():
      self.assertEqual(count, examples_per_cb)

  def test_generate_two_tone_set(self):
    """Test that the right way of combining tones and levels happens."""
    critical_bands = [0, 50, 100, 200, 500, 2000]
    masker_levels = [30, 80]
    probe_levels = [40]
    examples = data_generation.generate_two_tone_set(
        critical_bands,
        masker_levels=masker_levels,
        probe_levels=probe_levels,
        critical_bands_apart_probe=4,
        critical_bands_apart_masker=3,
        all_lower_probes=1,
        all_higher_probes=2)
    # expected_frequencies at middle of CB's maskers: 25, 350
    # expected level combinations: 40, 30 and 40, 80 (probe, masker)
    expected_examples = [(1250, 25, 40, 30),
                         (1250, 25, 40, 80),
                         (75, 25, 40, 30),
                         (75, 25, 40, 80),
                         (150, 25, 40, 30),
                         (150, 25, 40, 80),
                         (25, 350, 40, 30),
                         (25, 350, 40, 80),
                         (1250, 350, 40, 30),
                         (1250, 350, 40, 80),
                         (150, 350, 40, 30),
                         (150, 350, 40, 80)]
    expected_num_examples = len(expected_examples)
    self.assertEqual(len(examples), expected_num_examples)
    for i, example in enumerate(examples):
      example_unpacked = (int(example.probe.tone), int(example.masker.tone),
                          int(example.probe.level), int(example.masker.level))
      self.assertEqual(expected_examples[i], example_unpacked)

  def test_two_tone_set_unique(self):
    """Test that the examples are unique."""
    critical_bands = [0, 50, 100, 200, 500, 2000]
    masker_levels = [30, 40]
    probe_levels = [30, 40]
    examples = data_generation.generate_two_tone_set(
        critical_bands,
        masker_levels=masker_levels,
        probe_levels=probe_levels,
        critical_bands_apart_probe=2,
        critical_bands_apart_masker=3,
        all_lower_probes=1,
        all_higher_probes=2)
    expected_n_examples = len(set(examples))
    self.assertEqual(len(examples), expected_n_examples)

  def test_generate_data(self):
    examples_per_cb = 1000
    min_tones, max_tones = 2, 4
    desired_mean = 70
    desired_variance = 10
    clip_db = 90
    alpha = -4
    min_frequency, max_frequency = 20, 20000
    min_phons, max_phons = 0, 80
    critical_bands = [
        min_frequency, 100, 200, 300, 400, 505, 630, 770, 915, 1080, 1265, 1475,
        1720, 1990, 2310, 2690, 3125, 3675, 4350, 5250, 6350, 7650, 9400, 11750,
        15250, max_frequency
    ]
    all_examples = data_generation.generate_data(
        num_examples_per_cb=examples_per_cb,
        desired_mean=desired_mean,
        desired_variance=desired_variance,
        min_tones=min_tones,
        max_tones=max_tones,
        clip_db=clip_db,
        desired_skewness=alpha,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        critical_bands=critical_bands,
        min_phons=min_phons,
        max_phons=max_phons)
    covered_phons = []
    cb_counter = collections.Counter()
    for examples in all_examples.values():
      for example in examples:
        for freq_a, freq_b in itertools.combinations(example["frequencies"], 2):
          beat_range_a = data_generation.calculate_beat_range(freq_a)
          beat_range_b = data_generation.calculate_beat_range(freq_b)
          beat_range = max(beat_range_a, beat_range_b)
          self.assertGreaterEqual(abs(freq_a - freq_b), beat_range)
        for freq in example["frequencies"]:
          cb = data_generation.binary_search(critical_bands, freq) - 1
          cb_counter[cb] += 1
          self.assertGreaterEqual(freq, min_frequency)
          self.assertLessEqual(freq, max_frequency)
        for phons in example["phons"]:
          covered_phons.append(phons)
          self.assertGreaterEqual(phons, min_phons)
          self.assertLessEqual(phons, max_phons)
        for level in example["levels"]:
          self.assertGreaterEqual(level, 0)
          self.assertLessEqual(level, clip_db)
        self.assertGreaterEqual(len(example["frequencies"]), min_tones)
        self.assertLessEqual(len(example["frequencies"]), max_tones)
        self.assertEqual(len(example["frequencies"]), len(example["phons"]))
    self.assertAlmostEqual(np.round(np.mean(np.array(covered_phons))),
                           desired_mean)
    self.assertAlmostEqual(np.round(np.var(np.array(covered_phons))),
                           desired_variance)


if __name__ == "__main__":
  unittest.main()
