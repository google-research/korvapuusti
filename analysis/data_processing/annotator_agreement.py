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
"""Computes Krippendorff Alpha Annotater Agreement."""

import json
import os
from typing import Tuple, Dict

from absl import app
from absl import flags
import krippendorff
import numpy as np


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file_path",
    "data/parsed_answers.json",
    "JSON file with answers per annotator.")
flags.DEFINE_integer(
    "num_failing",
    50,
    "If an annotator hasn't annotated num_failing examples they will be "
    "excluded from the calculation.")


def get_reliability_matrix(
    input_file: str) -> Tuple[np.array, Dict[int, int], Dict[int, int]]:
  """Returns reliability matrix of answers per annotator."""
  with open(input_file, "r") as infile:
    data = json.load(infile)
    annotator_ids = data[0]["worker_ids"]
    annotator_to_idx = {annotator_ids[i]: i for i in range(len(annotator_ids))}
    annotator_failing_count = {annotator_ids[i]: 0 for i in range(
        len(annotator_ids))}
    num_annotators = len(annotator_ids)
    num_data_points = len(data)
    reliability_matrix = np.zeros([num_annotators, num_data_points])
    for i, data_point in enumerate(data):
      padded_answers = np.zeros(num_annotators) - 1  # TODO: change
      answers = np.array(data_point["perceived_probe_levels"])
      annotators = data_point["worker_ids"]
      for answer, annotator in zip(answers, annotators):
        annotator_idx = annotator_to_idx[annotator]
        padded_answers[annotator_idx] = answer
      failed = np.where(padded_answers == -1)[0]
      for fail in failed:
        annotator_failing_count[annotator_ids[fail]] += 1
      reliability_matrix[:, i] = padded_answers
  return reliability_matrix, annotator_to_idx, annotator_failing_count


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not os.path.exists(FLAGS.input_file_path):
    raise ValueError("No data found at %s" % FLAGS.input_file_path)

  reliability_matrix, annotators_to_idx, failed_count = get_reliability_matrix(
      FLAGS.input_file_path)

  alpha = krippendorff.alpha(reliability_matrix)
  print("Alpha without removing annotators: %.8f" % alpha)

  # Remmove annotator with too little answers.
  failed_annotators = []
  for annotator, count in failed_count.items():
    if count >= FLAGS.num_failing:
      row_to_remove = annotators_to_idx[annotator]
      failed_annotators.append(annotator)
      reliability_matrix = np.delete(reliability_matrix, row_to_remove, axis=0)

  alpha = krippendorff.alpha(reliability_matrix)
  print("Alpha with removing annotators: %.8f" % alpha)
  print("Removed annotators: ", failed_annotators)


if __name__ == "__main__":
  app.run(main)
