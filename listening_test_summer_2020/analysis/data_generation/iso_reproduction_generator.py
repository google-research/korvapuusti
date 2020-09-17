# Lint as: python3
"""Generate data for the listening tests to get saliency of subspectra of sound.
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
import os

from absl import app
from absl import flags

import data_analysis
import data_generation

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "save_directory",
    "listening_data",
    "Where to save the dataset and the plots.")
flags.DEFINE_integer(
    "num_examples_per_loudness_curve",
    8,
    "How many listening tests to generate for each ISO equal loudness curve.")
flags.DEFINE_integer(
    "num_curves",
    3,
    "How many equal loudness curves (i.e., phons level) to use.")
flags.DEFINE_integer(
    "clip_db",
    80,
    "At what SPL to clip the dB of the generated examples.")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  save_directory = os.path.join(
      os.path.dirname(os.path.realpath(__file__)),
      FLAGS.save_directory)
  if not os.path.exists(save_directory):
    os.mkdir(save_directory)

  data = data_generation.generate_iso_repro_examples(
      num_examples_per_curve=FLAGS.num_examples_per_loudness_curve,
      num_curves=FLAGS.num_curves,
      clip_db=FLAGS.clip_db)

  total_num_examples = data_analysis.save_iso_reproduction_examples(
      data, save_directory)

  data_analysis.plot_iso_examples(data, save_directory)


if __name__ == "__main__":
  app.run(main)
