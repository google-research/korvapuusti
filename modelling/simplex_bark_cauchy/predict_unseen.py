#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

from typing import List
import model
import dataset
import os
import numpy as np
from matplotlib import collections as matcoll
import matplotlib.pyplot as plt
from absl import app
import json
from absl import flags

FLAGS = flags.FLAGS


flags.DEFINE_string("vars_file", "all_vars_cauchy_run_1.txt",
                    "Where the logs are saved.")
flags.DEFINE_string("data_path", "extra_data",
                    "Data path.")
flags.DEFINE_string("infile_path",
                    "extra_data/extra_train_set.json",
                    "File path to data.")
flags.DEFINE_string("save_dir", "extra_data",
                    "Where the output should be saved.")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not os.path.exists(FLAGS.vars_file):
    raise ValueError("No data found at %s" % FLAGS.vars_file)

  if not os.path.exists(FLAGS.infile_path):
    raise ValueError("No data found at %s" % FLAGS.infile_path)

  if not os.path.exists(FLAGS.save_dir):
    os.mkdir(FLAGS.save_dir)

  with open(FLAGS.infile_path, "r") as infile:
    data = json.load(infile)

  model_class = model.FullModel()
  model_class.initialize_models(FLAGS.vars_file)
  total_error = 0
  total_points = 0
  for example in data:
    probe_frequency_khz = example["probe_frequency"] / 1000
    prediction = model_class.predict(example["masker_frequency"],
                                     example["probe_level"],
                                     example["masker_level"],
                                     probe_frequency_khz)
    predicted_perceived_level = example["probe_level"] - (prediction * 10)
    print("For masker f {}, masker level {}, probe f {}, probe level {}\n Predicted: {}".format(
        example["masker_frequency"], example["masker_level"],
        example["probe_frequency"], example["probe_level"],
        predicted_perceived_level))
    example["perceived_probe_levels"].append(int(predicted_perceived_level))
    total_points += 1

  with open(os.path.join(FLAGS.data_path, "predicted_extra_train_set.json"),
            "w") as outfile:
    json.dump(data, outfile, indent=4)

if __name__ == "__main__":
  app.run(main)
