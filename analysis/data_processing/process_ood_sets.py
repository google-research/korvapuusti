# Copyright 2020 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
"""Preprocesses data from listening tests."""

import json
import os
import random
from typing import Dict, List, Any, Tuple
import data_plotting
import data_helpers
import pprint
from scipy.io import wavfile

from absl import app
from absl import flags
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_path",
    "ood_sine",
    "JSON file with answers per annotator.")
flags.DEFINE_string(
    "input_file_pattern",
    "mask*",
    "The pattern that matches the files to read from data_path.")
flags.DEFINE_string(
    "output_directory",
    "ood_sine_sine",
    "Directory to save preprocessed data in.")
flags.DEFINE_string(
    "signal_shape",
    "sine",
    "Shape of the signal (sine or white).")
flags.DEFINE_string(
    "masker_frequencies",
    "843.0",
    "comma-seperated string of masker frequencies")
flags.DEFINE_string(
    "masker_levels",
    "50,70",
    "comma-seperated string of masker frequencies")
flags.DEFINE_string(
    "probe_levels",
    "20,40",
    "comma-seperated string of masker frequencies")
flags.DEFINE_integer("min_frequency", 20, "Minimum frequency for a tone.")
flags.DEFINE_integer("max_frequency", 20000, "Maximum frequency for a tone.")
flags.DEFINE_integer(
    "unity_decibel_level",
    100,
    "Which decibel level equals a sine at unity in the wavefiles."
    )


def process_example(raw_example: Dict[str, Any]) -> Tuple[Any]:
  probe_frequency = float(raw_example["ProbeFrequency"])
  probe_level = int(raw_example["ProbeLevel"])
  mask_frequencies = [float(frequency) for frequency in
                      raw_example["MaskFrequencies"]]
  masker_frequency = mask_frequencies[0]
  if len(mask_frequencies) > 1:
    raise ValueError("More than 1 mask frequency found: ",
                     raw_example["MaskFrequencies"])
  masker_level = [int(level) for level in raw_example["MaskLevels"]][0]
  perceived_probe_level = raw_example["ProbeDBSPLForEquivalentLoudness"]
  wavfile_identifier = "signal/mask.{}.wav".format(raw_example["EvaluationID"])
  return (probe_frequency, probe_level, masker_frequency, masker_level,
          perceived_probe_level, wavfile_identifier)


def process_evaluations_file(
    file_path: str, evaluations_file: str,
    frequencies_to_skip: float) -> Dict[str, Dict[str, Any]]:
  processed_data = {}
  with open(os.path.join(file_path, evaluations_file), "r") as infile:
    data = []
    for line in infile:
      data.append(json.loads(line))

    for raw_example in data:
      (probe_frequency, probe_level, masker_frequency, masker_level,
       perceived_probe_level, wavfile_identifier) = process_example(raw_example)
      wavfile_identifier = os.path.join(file_path, wavfile_identifier)
      if probe_frequency >= frequencies_to_skip:
        continue
      processed_data[probe_frequency] = {
          "probe_frequency": probe_frequency,
          "probe_level": probe_level,
          "perceived_probe_level": perceived_probe_level,
          "masker_frequency": masker_frequency,
          "masker_level": masker_level,
          "wavfile_identifier": wavfile_identifier
      }
  return processed_data


def compare_wavfiles(wavfile_paths: List[str]):
  processed_files = []
  for filepath in wavfile_paths:
    _, floats = wavfile.read(filepath)
    processed_files.append(floats)
  processed_files = np.array(processed_files)
  for row_i in processed_files:
    for row_j in processed_files:
      assert np.array_equal(row_i, row_j), ("Wavfiles not the same for paths: ",
                                            wavfile_paths)


def combine_evaluator_data(all_evaluator_data:
                           Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
                           all_masker_frequencies: List[float],
                           all_masker_levels: List[int], all_probe_levels: List[int]
                           ):
  combined_data = []
  curve_data_per_probe = {f: {l: [] for l in all_probe_levels} for f in all_masker_frequencies}
  for data_identifier, evaluators_data in all_evaluator_data.items():
    masker_frequency, mask_level, probe_level = [
        x for x in data_identifier.split("_")
    ]
    masker_frequency = float(masker_frequency)
    mask_level = int(mask_level)
    probe_level = int(probe_level)
    current_curve_data = {
        "masker_level": mask_level,
        "probe_frequencies": [],
        "probe_masking": [],
    }
    evaluations_per_probe = {}
    for evaluator_id, evaluator_data in evaluators_data.items():
      for probe_frequency, example in evaluator_data.items():
        if probe_frequency not in evaluations_per_probe:
          evaluations_per_probe[probe_frequency] = []
        evaluations_per_probe[probe_frequency].append(example)
    combined = []
    for probe_frequency, data in evaluations_per_probe.items():
      probe_frequencies = []
      probe_levels = []
      perceived_probe_levels = []
      masker_frequencies = []
      masker_levels = []
      wavfile_identifiers = []
      for example in data:
        probe_frequencies.append(example["probe_frequency"])
        probe_levels.append(example["probe_level"])
        perceived_probe_levels.append(example["perceived_probe_level"])
        masker_frequencies.append(example["masker_frequency"])
        masker_levels.append(example["masker_level"])
        wavfile_identifiers.append(example["wavfile_identifier"])
      assert len(set(probe_frequencies)) == 1, ("Different probes in combining: ", data)
      assert len(
          set(probe_levels)) == 1, "Different probes levels in combining."
      assert len(
          set(masker_frequencies)) == 1, "Different maskers in combining."
      assert len(
          set(masker_levels)) == 1, "Different masker levels in combining."
      assert probe_levels[0] == probe_level, "Wrong match of probe levels"
      assert masker_levels[0] == mask_level, "Wrong match of masker levels"
      compare_wavfiles(wavfile_identifiers)
      current_curve_data["probe_frequencies"].append(probe_frequencies[0])
      current_curve_data["probe_masking"].append(0)
      combined_data.append({
          "probe_frequency": probe_frequencies[0],
          "probe_level": probe_levels[0],
          "perceived_probe_levels": perceived_probe_levels,
          "worker_ids": [],
          "masker_frequency": masker_frequencies[0],
          "masker_level": masker_levels[0],
          "wavfile_identifier": wavfile_identifiers[0]
      })
    curve_data_per_probe[masker_frequency][probe_level].append(current_curve_data)
  curve_data = []
  for masker_frequency in all_masker_frequencies:
    for probe_level in all_probe_levels:
      print(masker_frequency, probe_level)
      curve_data.append({"masker_frequency": masker_frequency,
                       "probe_level": probe_level,
                       "curves": curve_data_per_probe[masker_frequency][probe_level]})
  return combined_data, curve_data


def prepare_data_modeling(data: List[Dict[str, Any]],
                          curve_data: List[Dict[str, Any]],
                          save_directory):
  lookup_table = data_helpers.AnswerLookupTable()
  for example in data:
    lookup_table.add(example["masker_frequency"], example["probe_level"],
                     example["masker_level"], example["probe_frequency"],
                     example)
  test_set = []
  answers_matched = 0
  for i, masker_probe_curves in enumerate(curve_data):
    masker_frequency = float(masker_probe_curves["masker_frequency"])
    probe_level = int(masker_probe_curves["probe_level"])
    curves = masker_probe_curves["curves"]
    for j, curve in enumerate(curves):
      curve_data[i]["curves"][j]["failed"] = []
      masker_level = int(curve["masker_level"])
      probe_frequencies = curve["probe_frequencies"]
      for k, probe_frequency in enumerate(probe_frequencies):
        probe_frequency = float(probe_frequency)
        example_answers = lookup_table.extract(masker_frequency, probe_level,
                                               masker_level, probe_frequency)
        if example_answers:
          answers = example_answers["perceived_probe_levels"]
          perceived_levels = np.array(answers)
          masking = probe_level - perceived_levels
          masking[masking < 0] = 0
          curve_data[i]["curves"][j]["probe_masking"][k] = list(masking)
          answers_matched += 1
          test_set.append(example_answers)
        else:
          curve_data[i]["curves"][j]["failed"].append(k)
  data_plotting.plot_masking_patterns(curve_data,
                                      save_directory=save_directory)
  with open(os.path.join(save_directory, "preprocessed_ood_test_set.json"),
            "w") as outfile:
    json.dump(test_set, outfile, indent=4)
  return

def get_data(data_directory: str, shape: str, masker_frequencies: List[float],
             masker_levels: List[int], probe_levels: List[int]):

  # List of hardcoded frequencies to skip per evaluator
  skip_frequencies_per_evaluator = [9400, 10000, 10000]

  num_evaluators = 3
  data_evaluators = {}

  for i in range(num_evaluators):
    evaluator_dir = "evaluator_{}".format(i + 1)
    path = os.path.join(data_directory, evaluator_dir)
    for masker_frequency in masker_frequencies:
      for mask_level in masker_levels:
        for probe_level in probe_levels:
          data_identifier = "{}_{}_{}".format(masker_frequency,
                                              mask_level, probe_level)
          if data_identifier not in data_evaluators:
            data_evaluators[data_identifier] = {}
          masker_dir = "mask_{}Hz_{}dB_probe_{}dB_{}".format(
              int(masker_frequency), int(mask_level), int(probe_level),
              shape)
          final_path = os.path.join(path, masker_dir)
          evaluations = process_evaluations_file(
              final_path, "evaluations.json", skip_frequencies_per_evaluator[i])
          data_evaluators[data_identifier]["evaluator_{}".format(i)] = evaluations
  combined_data, curve_data = combine_evaluator_data(data_evaluators,
                                                     masker_frequencies,
                                                     masker_levels,
                                                     probe_levels)
  return combined_data, curve_data


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not os.path.exists(FLAGS.data_path):
    raise ValueError("No data found at %s" % FLAGS.data_path)

  if not os.path.exists(FLAGS.output_directory):
    os.mkdir(FLAGS.output_directory)

  masker_frequencies = [float(f) for f in FLAGS.masker_frequencies.split(",")]
  masker_levels = [int(l) for l in FLAGS.masker_levels.split(",")]
  probe_levels = [int(l) for l in FLAGS.probe_levels.split(",")]
  data, curve_data = get_data(FLAGS.data_path, FLAGS.signal_shape,
                              masker_frequencies,
                              masker_levels,
                              probe_levels)

  with open(os.path.join(FLAGS.output_directory,
                         "ood_test_{}.json".format(FLAGS.signal_shape)),
            "w") as outfile:
    json.dump(data, outfile, indent=4)

  prepare_data_modeling(data, curve_data, FLAGS.output_directory)

if __name__ == "__main__":
  app.run(main)
