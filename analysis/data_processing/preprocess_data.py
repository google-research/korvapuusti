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

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
flags.DEFINE_float(
    "percentage_test",
    0.1,
    "Percentage of data to make into a test set.")
flags.DEFINE_integer(
    "seed",
    1,
    "Random seed."
    )
flags.DEFINE_integer("min_frequency", 20, "Minimum frequency for a tone.")
flags.DEFINE_integer("max_frequency", 20000, "Maximum frequency for a tone.")
flags.DEFINE_integer(
    "unity_decibel_level",
    90,
    "Which decibel level equals a sine at unity in the wavefiles."
    )


class AnswerLookupTable():
  """Lookup Table for finding perceived probe levels from CC answers."""

  def __init__(self):
    self.table = {}

  def get_table_key(self, masker_frequency: float,
                    probe_level: int,
                    masker_level: int):
    return "{},{},{}".format(str(masker_frequency),
                             str(probe_level),
                             str(masker_level))

  def add(self, masker_frequency: float, probe_level: int, masker_level: int,
          probe_frequency: float, answers: List[int]):
    table_key = self.get_table_key(masker_frequency, probe_level, masker_level)
    if table_key not in self.table.keys():
      self.table[table_key] = {}
    self.table[table_key][probe_frequency] = answers

  def extract(self, masker_frequency: float, probe_level: int,
              masker_level: int, probe_frequency: float):
    table_key = self.get_table_key(masker_frequency, probe_level, masker_level)
    if self.table.get(table_key):
      return self.table[table_key].get(probe_frequency)
    else:
      return None


def plot_histogram(values: List[Any], path: str, bins=None, logscale=False,
                   x_label="", title=""):
  """Plots and saves a histogram."""
  _, ax = plt.subplots(1, 1)
  if not bins:
    ax.hist(values, histtype="stepfilled", alpha=0.2)
  else:
    ax.hist(values, bins=bins, histtype="stepfilled", alpha=0.2)
  if logscale:
    plt.xscale("log")
  ax.set_ylabel("Occurrence Count")
  ax.set_xlabel(x_label)
  ax.set_title(title)
  plt.savefig(path)


def plot_heatmap(data: np.ndarray, path: str):
  """Plots and saves a heatmap of critical band co-occurrences."""
  fig, ax = plt.subplots()
  fig.set_size_inches(18.5, 18.5)
  # We want to show all ticks...
  ax.set_xticks(np.arange(data.shape[1]))
  ax.set_yticks(np.arange(data.shape[0]))
  # ... and label them with the respective list entries.
  labels = ["CB {}".format(i) for i in range(data.shape[1])]
  ax.set_xticklabels(labels)
  ax.set_yticklabels(labels)
  ax.set_title("Count per combination of Critical Bands in Examples")
  sns.heatmap(data, annot=True, fmt="d", xticklabels=labels, yticklabels=labels,
              ax=ax)
  plt.savefig(path)


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
    for example in data:
      masker_probe_tuple = (example["masker_frequency"],
                            example["probe_frequency"])
      if masker_probe_tuple not in masker_probe_frequencies_to_remove:
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
    plot_heatmap(cb_combinations, os.path.join(save_directory,
                                               "new_heatmap.png"))
    return cleaned_data


def drop_too_high_variance(data: List[Dict[str, Any]],
                           save_directory: str) -> List[Dict[str, Any]]:
  variances = []
  for answer in data:
    perceived_probe_levels = np.array(answer["perceived_probe_levels"])
    variance = np.std(perceived_probe_levels)
    variances.append(variance)
  plot_histogram(variances, os.path.join(save_directory, "variances.png"))
  quantile = np.quantile(np.array(variances), 0.9)
  cleaned_data = []
  cleaned_variances = []
  num_dropped = 0
  for answer in data:
    perceived_probe_levels = np.array(answer["perceived_probe_levels"])
    variance = np.std(perceived_probe_levels)
    if variance <= quantile:
      cleaned_data.append(answer)
      cleaned_variances.append(variance)
    else:
      num_dropped += 1
  plot_histogram(cleaned_variances, os.path.join(save_directory,
                                                 "cleaned_variances.png"))
  print("Num dropped %d" % num_dropped)
  return cleaned_data


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


def plot_masking_patterns(curves: List[Dict[str, Any]],
                          save_directory: str):
  """Plots the data in Zwicker-style."""
  for masker_freq_probe_level in curves:
    masker_frequency = masker_freq_probe_level["masker_frequency"]
    probe_level = masker_freq_probe_level["probe_level"]
    _, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_prop_cycle(color=[
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
        "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
        "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
        "#17becf", "#9edae5"
    ])
    plt.xscale("log")
    ax.set_xlabel("Probe Frequency (Hz)")
    ax.set_ylabel("Masked SPL (dB)")
    ax.set_title("Masker Frequency {}, Probe Level {}".format(
        masker_frequency, probe_level))
    plt.axvline(x=masker_frequency)
    plt.axhline(y=probe_level)
    current_levels = []
    mean_masking_per_level = []
    for masker_curve in masker_freq_probe_level["curves"]:
      masker_level = masker_curve["masker_level"]
      current_levels.append(masker_level)
      frequencies = masker_curve["probe_frequencies"]
      masking = masker_curve["probe_masking"]
      average_masking = [np.mean(np.array(m)) for m in masking]
      mean_masking_per_level.append(average_masking)
      std_masking = [np.std(np.array(m)) for m in masking]
      plt.errorbar(
          frequencies,
          average_masking,
          yerr=std_masking,
          fmt="o",
          label="Masker Level {}".format(masker_level))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    save_file = "masker_{}_probe_{}".format(masker_frequency, probe_level)
    with open(os.path.join(save_directory, save_file + ".txt"), "wt") as infile:
      for level, mean_masking in zip(current_levels, mean_masking_per_level):
        infile.write("level {}: ".format(level))
        freq_maskers_str = ";".join(["{},{}".format(f, m) for f, m in zip(
            frequencies, mean_masking)])
        infile.write(freq_maskers_str)
        infile.write("\n")
    plt.savefig(os.path.join(save_directory,
                             save_file + ".png"))


def prepare_data_modeling(train_set: List[Dict[str, Any]], curves_file: str,
                          save_directory: str):
  lookup_table = AnswerLookupTable()
  for example in train_set:
    lookup_table.add(example["masker_frequency"], example["probe_level"],
                     example["masker_level"], example["probe_frequency"],
                     example["perceived_probe_levels"])
  with open(curves_file, "r") as infile:
    answers_matched = 0
    curve_data = json.load(infile)
    for i, masker_probe_curves in enumerate(curve_data):
      masker_frequency = float(masker_probe_curves["masker_frequency"])
      probe_level = int(masker_probe_curves["probe_level"])
      curves = masker_probe_curves["curves"]
      for j, curve in enumerate(curves):
        masker_level = int(curve["masker_level"])
        probe_frequencies = curve["probe_frequencies"]
        for k, probe_frequency in enumerate(probe_frequencies):
          probe_frequency = float(probe_frequency)

          answers = lookup_table.extract(masker_frequency, probe_level,
                                         masker_level, probe_frequency)
          if answers:
            perceived_levels = np.array(answers)
            masking = probe_level - perceived_levels
            masking[masking < 0] = 0
            curve_data[i]["curves"][j]["probe_masking"][k] = masking
            answers_matched += 1
    plot_masking_patterns(curve_data, save_directory=save_directory)
  return answers_matched


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not os.path.exists(FLAGS.input_file_path):
    raise ValueError("No data found at %s" % FLAGS.input_file_path)

  if not os.path.exists(FLAGS.output_directory):
    os.mkdir(FLAGS.output_directory)

  critical_bands = [
      FLAGS.min_frequency, 100, 200, 300, 400, 505, 630, 770, 915, 1080, 1265,
      1475, 1720, 1990, 2310, 2690, 3125, 3675, 4350, 5250, 6350, 7650, 9400,
      11750, 15250, FLAGS.max_frequency
  ]

  random.seed(FLAGS.seed)

  data = get_data(FLAGS.input_file_path, FLAGS.specs_file_path,
                  FLAGS.output_directory, critical_bands,
                  FLAGS.unity_decibel_level)
  data_cleaned = drop_too_high_variance(data, FLAGS.output_directory)
  train_set, test_set = split_data(data_cleaned, FLAGS.percentage_test)
  with open(os.path.join(FLAGS.output_directory, "train_set.json"),
            "w") as outfile:
    json.dump(train_set, outfile, indent=4)
  with open(os.path.join(FLAGS.output_directory, "test_set.json"),
            "w") as outfile:
    json.dump(test_set, outfile, indent=4)

  answers_matched = prepare_data_modeling(train_set, FLAGS.specs_file_path,
                                          FLAGS.output_directory)

  print("Answers matched to curve: %d" % answers_matched)


if __name__ == "__main__":
  app.run(main)
