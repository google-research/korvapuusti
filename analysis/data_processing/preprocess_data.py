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

from absl import app
from absl import flags
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_path",
    "data",
    "JSON file with answers per annotator.")
flags.DEFINE_string(
    "input_file_path",
    "data/parsed_answers.json",
    "JSON file with answers per annotator.")
flags.DEFINE_string(
    "specs_file_path",
    "data/SPECS_probes_two_tone_set.json",
    "JSON file with answers per annotator.")
flags.DEFINE_string(
    "output_directory",
    "output",
    "Directory to save preprocessed data in.")
flags.DEFINE_string(
    "test_directory",
    "testdata",
    "Directory to save preprocessed data in.")
flags.DEFINE_float(
    "percentage_test",
    0.15,
    "Percentage of data to make into a test set.")
flags.DEFINE_integer(
    "seed",
    2,
    "Random seed."
    )
flags.DEFINE_integer("min_frequency", 20, "Minimum frequency for a tone.")
flags.DEFINE_integer("max_frequency", 20000, "Maximum frequency for a tone.")
flags.DEFINE_integer(
    "unity_decibel_level",
    90,
    "Which decibel level equals a sine at unity in the wavefiles."
    )


def binary_search(arr, item):
  """Finds closest index to the left of an item in arr."""
  low = 0
  high = len(arr) - 1
  mid = 0

  while low <= high:
    mid = (high + low) // 2
    # Check if item is present at mid
    if arr[mid] < item:
      low = mid
    # If item is greater, ignore left half
    elif arr[mid] > item:
      high = mid
    # If item is smaller, ignore right half
    else:
      return mid
    if arr[high] <= item:
      return high

    if arr[low] <= item < arr[low + 1]:
      return low
  return mid


def string_from_frequency(frequency: float) -> str:
  str_frequency = str(frequency) + "0Hz"
  num_chars = 10
  num_leading_zeros = num_chars - len(str_frequency)
  return "0" * num_leading_zeros + str_frequency


def make_wavfile_name(probe_frequency: float, masker_frequency: float,
                      probe_level: int, masker_level: int,
                      unity_db_level=90) -> str:
  probe_str = string_from_frequency(probe_frequency)
  masker_str = string_from_frequency(masker_frequency)
  probe_level_normalized = probe_level - unity_db_level
  masker_level_normalized = masker_level - unity_db_level
  probe_level_str = str(probe_level_normalized) + ".00dBFS"
  masker_level_str = str(masker_level_normalized) + ".00dBFS"
  wavfile_name = "combined_{}_{}+{}_{}".format(masker_str, masker_level_str,
                                               probe_str, probe_level_str)
  return wavfile_name


def get_data(
    input_file: str, specs_file: str, save_directory: str,
    critical_bands: List[int], unity_db_level: int) -> List[Dict[str, Any]]:
  """Returns data."""
  with open(input_file, "r") as infile:
    data = json.load(infile)
    print("Initial num examples: {}".format(len(data)))

    # Make sure that the data reflects 4 CBs below and above the masker
    # Instead of 8 below and 4 above
    # Due to mistake in data generation
    masker_probe_frequencies_to_remove = set()
    cb_combinations = np.zeros(
      [len(critical_bands) - 1,
       len(critical_bands) - 1], dtype=int)
    with open(specs_file, "r") as infile:
      specs = json.load(infile)
      for masker_frequency_curves in specs:
        masker_frequency = masker_frequency_curves["masker_frequency"]
        masker_cb = binary_search(critical_bands, masker_frequency)
        for curve in masker_frequency_curves["curves"]:
          cbs = []
          for probe_frequency in curve["probe_frequencies"]:
            cb_probe = binary_search(critical_bands, probe_frequency)
            cbs.append(cb_probe)
          cbs.sort()
          if masker_cb < cbs[0]:
            continue
          closest_idx = binary_search(cbs, masker_cb)
          probe_frequencies = curve["probe_frequencies"]
          probe_frequencies.sort()
          cbs_to_remove = [closest_idx - 4, closest_idx - 6]
          masker_probe_frequencies_to_remove.add(
              (masker_frequency, probe_frequencies[closest_idx - 4]))
          masker_probe_frequencies_to_remove.add(
              (masker_frequency, probe_frequencies[closest_idx - 6]))
    cleaned_data = []
    with open("data/redone_parsed_answers.json", "r") as infile2:
      redone_data = json.load(infile2)
      num_redone = 0
      before_levels = []
      after_levels = []
      for example in data:
        masker_probe_tuple = (example["masker_frequency"],
                              example["probe_frequency"])
        if masker_probe_tuple not in masker_probe_frequencies_to_remove:

          # Replace some of the answers that went wrong with re-done answers.
          for redone_example in redone_data:
            if (example["probe_frequency"] == redone_example["probe_frequency"]
                and example["masker_frequency"] ==
                redone_example["masker_frequency"] and
                example["probe_level"] == redone_example["probe_level"] and
                example["masker_level"] == redone_example["masker_level"]):
              num_redone += 1
              correct_idx_annotator = example["worker_ids"].index(
                  redone_example["worker_ids"][0])
              before_levels.append(
                  example["perceived_probe_levels"][correct_idx_annotator])
              after_levels.append(redone_example["perceived_probe_levels"][0])
              example["perceived_probe_levels"][
                  correct_idx_annotator] = redone_example[
                      "perceived_probe_levels"][0]
          cleaned_data.append(example)
          cb_probe = binary_search(critical_bands, example["probe_frequency"])
          cb_masker = binary_search(critical_bands, example["masker_frequency"])
          cb_combinations[cb_masker, cb_probe] += 1
          wavefile_name = make_wavfile_name(example["probe_frequency"],
                                            example["masker_frequency"],
                                            example["probe_level"],
                                            example["masker_level"],
                                            unity_db_level)
          example["wavfile_identifier"] = wavefile_name
    data_plotting.plot_heatmap(cb_combinations,
                               os.path.join(save_directory, "new_heatmap.png"))
    return cleaned_data


def drop_too_high_variance(data: List[Dict[str, Any]],
                           save_directory: str) -> List[Dict[str, Any]]:
  variances = []
  for answer in data:
    perceived_probe_levels = np.array(answer["perceived_probe_levels"])
    variance = np.std(perceived_probe_levels)
    variances.append(variance)
  data_plotting.plot_histogram(variances,
                               os.path.join(save_directory, "variances.png"))
  quantile = np.quantile(np.array(variances), 0.85)
  cleaned_data = []
  cleaned_variances = []
  num_dropped = 0
  other_variances = []
  variances_temp = []
  answers_temp = []
  redo_examples = []
  for i, answer in enumerate(data):
    perceived_probe_levels = np.array(answer["perceived_probe_levels"])
    variance = np.std(perceived_probe_levels)
    if 160 < i < 200:
      variances_temp.append(variance)
      redo_examples.append("[[{},{}],[{},{}]]".format(
          int(answer["masker_frequency"]), int(answer["masker_level"]),
          int(answer["probe_frequency"]), int(answer["probe_level"])))
      answers_temp.append(perceived_probe_levels)
    else:
      other_variances.append(variance)
    if variance <= quantile:
      cleaned_data.append(answer)
      cleaned_variances.append(variance)
    else:
      num_dropped += 1
  data_plotting.plot_histogram(cleaned_variances,
                               os.path.join(save_directory,
                                            "cleaned_variances.png"))
  print("Num dropped %d" % num_dropped)
  return cleaned_data, redo_examples


def split_data(
    data: List[Dict[str, Any]], percentage_test: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
  num_examples = len(data)
  num_test = percentage_test * num_examples
  indices = [i for i in range(num_examples)]
  test_set_idx = random.sample(indices, int(num_test))
  test_set_idx = set(test_set_idx)
  train_set = []
  test_set = []
  for i, example in enumerate(data):
    if i in test_set_idx:
      test_set.append(example)
    else:
      train_set.append(example)
  print("Num train: %d, Num test: %d" % (len(train_set), len(test_set)))
  return train_set, test_set


def prepare_data_modeling(train_set: List[Dict[str, Any]], curves_file: str,
                          save_directory: str):
  lookup_table = data_helpers.AnswerLookupTable()
  for example in train_set:
    lookup_table.add(example["masker_frequency"], example["probe_level"],
                     example["masker_level"], example["probe_frequency"],
                     example)
  preprocessed_train_set = []
  with open(curves_file, "r") as infile:
    answers_matched = 0
    curve_data = json.load(infile)
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
            # Hardcoded removal of failed probes (too high frequency).
            if probe_frequency == 17625.0:
              curve_data[i]["curves"][j]["failed"].append(k)
            else:
              masking = probe_level - perceived_levels
              masking[masking < 0] = 0
              curve_data[i]["curves"][j]["probe_masking"][k] = list(masking)
              answers_matched += 1
              preprocessed_train_set.append(example_answers)
          else:
            curve_data[i]["curves"][j]["failed"].append(k)
    data_plotting.plot_masking_patterns_grid(curve_data,
                                             save_directory=save_directory)
    data_plotting.plot_masking_patterns(curve_data,
                                        save_directory=save_directory)
    with open(os.path.join(save_directory, "preprocessed_train_set.json"),
              "w") as outfile:
      json.dump(preprocessed_train_set, outfile, indent=4)
  return answers_matched


def prepare_data_test(test_set: List[Dict[str, Any]], curves_file: str,
                      save_directory: str):
  lookup_table = data_helpers.AnswerLookupTable()
  for example in test_set:
    lookup_table.add(example["masker_frequency"], example["probe_level"],
                     example["masker_level"], example["probe_frequency"],
                     example)
  test_set = []
  with open(curves_file, "r") as infile:
    answers_matched = 0
    curve_data = json.load(infile)
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
    data_plotting.plot_masking_patterns_grid(curve_data,
                                             save_directory=save_directory)
    data_plotting.plot_masking_patterns(curve_data,
                                        save_directory=save_directory)
    with open(os.path.join(save_directory, "preprocessed_test_set.json"),
              "w") as outfile:
      print("Len test set: ", len(test_set))
      json.dump(test_set, outfile, indent=4)
  return answers_matched


def prepare_csv(redo_examples: List[str], csv_path: str, csv_outpath: str):
  unique_examples = set(redo_examples)
  with open(csv_path, "r") as infile:
    newlines = []
    for i, line in enumerate(infile.readlines()):
      if i == 0:
        newlines.append(line)
        continue
      splitted_line = line.split('",')
      combined_tone = str(splitted_line[1].strip('"'))
      if combined_tone in redo_examples:
        newlines.append(line)
    with open(csv_outpath, "w") as outfile:
      for line in newlines:
        outfile.write(line)
  return


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not os.path.exists(FLAGS.input_file_path):
    raise ValueError("No data found at %s" % FLAGS.input_file_path)

  if not os.path.exists(FLAGS.output_directory):
    os.mkdir(FLAGS.output_directory)

  if not os.path.exists(FLAGS.test_directory):
    os.mkdir(FLAGS.test_directory)

  critical_bands = [
      FLAGS.min_frequency, 100, 200, 300, 400, 505, 630, 770, 915, 1080, 1265,
      1475, 1720, 1990, 2310, 2690, 3125, 3675, 4350, 5250, 6350, 7650, 9400,
      11750, 15250, FLAGS.max_frequency
  ]

  random.seed(FLAGS.seed)

  data = get_data(FLAGS.input_file_path, FLAGS.specs_file_path,
                  FLAGS.output_directory, critical_bands,
                  FLAGS.unity_decibel_level)
  data_cleaned, redo_examples = drop_too_high_variance(data,
                                                       FLAGS.output_directory)
  prepare_csv(redo_examples, "output/probes_two_tone_set.csv",
              "output/redo_probes_two_tone_set.csv")
  train_set, test_set = split_data(data_cleaned, FLAGS.percentage_test)
  with open(os.path.join(FLAGS.output_directory, "train_set.json"),
            "w") as outfile:
    json.dump(train_set, outfile, indent=4)
  with open(os.path.join(FLAGS.output_directory, "test_set.json"),
            "w") as outfile:
    json.dump(test_set, outfile, indent=4)

  answers_matched = prepare_data_modeling(train_set, FLAGS.specs_file_path,
                                          FLAGS.output_directory)
  answers_matched_test = prepare_data_test(test_set, FLAGS.specs_file_path,
                                           FLAGS.test_directory)

  print("Answers matched to curve train: %d" % answers_matched)
  print("Answers matched to curve test: %d" % answers_matched_test)


if __name__ == "__main__":
  app.run(main)
