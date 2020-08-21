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
import csv

from absl import app
from absl import flags
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file_path",
    "extra_data/probes_two_tone_set_extra_data.csv",
    "JSON file with answers per annotator.")
flags.DEFINE_string(
    "output_directory",
    "extra_data",
    "Directory to save preprocessed data in.")
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


def get_data(
    input_file: str, save_directory: str,
    critical_bands: List[int], unity_db_level: int) -> List[Dict[str, Any]]:
  """Returns data."""
  with open(input_file, "r") as infile:
    csvreader = csv.reader(infile, delimiter=',')
    data = []
    for i, raw_example_line in enumerate(csvreader):
      if i == 0:
        continue

      example_specs = raw_example_line[2].split("],[")
      masker_frequency, masker_level = example_specs[0].strip("[[").split(",")
      probe_frequency, probe_level = example_specs[1].strip("]]").split(",")

      wavfile_identifier = raw_example_line[4].split("/")[-1]
      example = {
          "probe_frequency": float(probe_frequency),
          "probe_level": int(probe_level),
          "perceived_probe_levels": [],
          "worker_ids": [],
          "masker_frequency": float(masker_frequency),
          "masker_level": int(masker_level),
          "wavfile_identifier": wavfile_identifier
      }
      data.append(example)
    return data


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

  data = get_data(FLAGS.input_file_path,
                  FLAGS.output_directory, critical_bands,
                  FLAGS.unity_decibel_level)

  with open(os.path.join(FLAGS.output_directory, "extra_train_set.json"),
            "w") as outfile:
    json.dump(data, outfile, indent=4)


if __name__ == "__main__":
  app.run(main)
