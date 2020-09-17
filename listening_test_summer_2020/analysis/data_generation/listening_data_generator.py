# Lint as: python3
"""Generate data for the listening tests to get saliency of subspectra of sound.

TODO: visualize more of the data (distribution of levels per example)
TODO: randomize order of writing data
TODO: beatrange of +/- 50 because non distinguishable is no problem

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
    "examples_per_critical_band", 300,
    "Number of examples to generate per frequency bin as a probe.")
flags.DEFINE_integer("min_tones", 2,
                     "Minimum number of frequencies in each example.")
flags.DEFINE_integer("max_tones", 4,
                     "Maximum number of frequencies in each example")
flags.DEFINE_integer("min_phons", 0, "Minimum phons for each tone level.")
flags.DEFINE_integer("max_phons", 80, "Maximum phons for each tone level.")
flags.DEFINE_integer("min_frequency", 20, "Minimum frequency for a tone.")
flags.DEFINE_integer("max_frequency", 20000, "Maximum frequency for a tone.")
flags.DEFINE_integer("clip_db", 90,
                     "At what level to clip the decibel of each tone.")
flags.DEFINE_integer(
    "skewness_parameter", -4,
    "Skewness of skew Normal distribution to sample phons from.")
flags.DEFINE_integer("log_base", 2,
                     "Base of log scale to sample frequencies from.")
flags.DEFINE_integer(
    "desired_mean", 60,
    "Desired mean for the skew Normal distribution to sample phons from.")
flags.DEFINE_integer(
    "desired_variance", 140,
    "Desired variance for the skew Normal distribution to sample phons from.")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  assert FLAGS.min_frequency < 100, "--min_frequency higher than 100."
  assert FLAGS.max_frequency > 15250, "--max_frequency lower than 15250."

  save_directory = os.path.join(
      os.path.dirname(os.path.realpath(__file__)),
      FLAGS.save_directory)
  if not os.path.exists(save_directory):
    os.mkdir(save_directory)

  # TODO(lauraruis): define this elsewhere
  critical_bands = [
      FLAGS.min_frequency, 100, 200, 300, 400, 505, 630, 770, 915, 1080, 1265,
      1475, 1720, 1990, 2310, 2690, 3125, 3675, 4350, 5250, 6350, 7650, 9400,
      11750, 15250, FLAGS.max_frequency
  ]

  # Generate the data.
  data = data_generation.generate_data(
      num_examples_per_cb=FLAGS.examples_per_critical_band,
      desired_mean=FLAGS.desired_mean,
      desired_variance=FLAGS.desired_variance,
      min_tones=FLAGS.min_tones,
      max_tones=FLAGS.max_tones,
      clip_db=FLAGS.clip_db,
      desired_skewness=FLAGS.skewness_parameter,
      min_frequency=FLAGS.min_frequency,
      max_frequency=FLAGS.max_frequency,
      critical_bands=critical_bands,
      min_phons=FLAGS.min_phons,
      max_phons=FLAGS.max_phons)

  # Run and save some analysis on the generated data.
  (covered_num_tones_per_cb, total_unique_examples,
   total_num_examples_listeners) = data_analysis.save_data(
       data, save_directory, critical_bands)


if __name__ == "__main__":
  app.run(main)
