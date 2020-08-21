# Lint as: python3
"""Tests for data_analysis.

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
import csv
import json
import os
import unittest

from absl.testing import absltest
import data_analysis
import data_generation

class DataAnalysisTest(unittest.TestCase):

  def setUp(self):
    """Sets up directory."""
    super(DataAnalysisTest, self).setUp()
    self.save_directory = "testdir"
    if not os.path.exists(self.save_directory):
      os.mkdir(self.save_directory)

  def test_find_frequency_bin_low_cornercase(self):
    frequency = 21
    expected_bin = 1
    actual_bin = data_analysis.find_stft_bin(frequency)
    self.assertEqual(expected_bin, actual_bin)

  def test_find_frequency_bin_high_cornercase(self):
    frequency = 20000
    expected_bin = 929
    actual_bin = data_analysis.find_stft_bin(frequency)
    self.assertEqual(expected_bin, actual_bin)

  def test_find_frequency_bin_middlecase(self):
    frequency = 119
    expected_bin = 6
    actual_bin = data_analysis.find_stft_bin(frequency)
    self.assertEqual(expected_bin, actual_bin)

  def test_save_two_tone_set(self):
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

    # [[masker_frequency, masker_level], [probe_frequency, probe level]]
    expected_examples_probes = {
        "[[25.0,30],[1250.0,40]]", "[[25.0,80],[1250.0,40]]",
        "[[25.0,30],[75.0,40]]", "[[25.0,80],[75.0,40]]",
        "[[25.0,30],[150.0,40]]", "[[25.0,80],[150.0,40]]",
        "[[350.0,30],[25.0,40]]", "[[350.0,80],[25.0,40]]",
        "[[350.0,30],[1250.0,40]]", "[[350.0,80],[1250.0,40]]",
        "[[350.0,30],[150.0,40]]", "[[350.0,80],[150.0,40]]"
    }
    expected_examples_maskers = expected_examples_probes.copy()

    expected_curves_probes_1 = {
        "masker_frequency":
            25.0,
        "probe_level":
            40,
        "curves": [{
            "masker_level": 30,
            "probe_frequencies": [1250.0, 75.0, 150.0],
            "probe_masking": [[0], [0], [0]]
        }, {
            "masker_level": 80,
            "probe_frequencies": [1250.0, 75.0, 150.0],
            "probe_masking": [[0], [0], [0]]
        }]
    }
    expected_curves_probes_2 = {
        "masker_frequency":
            350.0,
        "probe_level":
            40,
        "curves": [{
            "masker_level": 30,
            "probe_frequencies": [25.0, 1250.0, 150.0],
            "probe_masking": [[0], [0], [0]]
        }, {
            "masker_level": 80,
            "probe_frequencies": [25.0, 1250.0, 150.0],
            "probe_masking": [[0], [0], [0]]
        }]
    }

    expected_curves = [expected_curves_probes_1, expected_curves_probes_2]

    # Run and save some analysis on the generated data.
    (probes_path, probes_specs_path, maskers_path,
     _) = data_analysis.save_two_tone_set(
         examples, {}, critical_bands, self.save_directory)

    with open(
        probes_path,
        "r") as probes_infile:
      probes_reader = csv.reader(probes_infile, delimiter=",")
      with open(
          maskers_path,
          "r") as maskers_infile:
        maskers_reader = csv.reader(maskers_infile, delimiter=",")
        probes_probes = set()
        probes_maskers = set()
        for probe_example, masker_example in zip(probes_reader, maskers_reader):
          if probe_example[0] == "id":
            continue
          probe_probe = probe_example[1]
          probe_masker = masker_example[1]
          probes_probes.add(probe_probe)
          probes_maskers.add(probe_masker)
          self.assertIn(probe_example[2], expected_examples_probes)
          expected_examples_probes.remove(probe_example[2])
          self.assertIn(masker_example[2], expected_examples_maskers)
          expected_examples_maskers.remove(masker_example[2])
          self.assertIn(probe_probe, probe_example[2])
          self.assertIn(probe_masker, masker_example[2])

        # Check that the set of probes and maskers are disjunct.
        num_probes = len(probes_probes)
        num_maskers = len(probes_maskers)
        probes_probes.update(probes_maskers)
        self.assertEqual(len(probes_probes), num_probes + num_maskers)
    self.assertEqual(len(expected_examples_probes), 0)
    self.assertEqual(len(expected_examples_maskers), 0)

    with open(probes_specs_path, "r") as infile:
      probes_specs = json.load(infile)
      for i, curve in enumerate(probes_specs):
        self.assertEqual(curve, expected_curves[i])


if __name__ == "__main__":
  unittest.main()
