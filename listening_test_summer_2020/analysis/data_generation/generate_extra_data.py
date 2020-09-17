"""TODO(lauraruis): DO NOT SUBMIT without one-line documentation for generate_extra_data.

TODO(lauraruis): DO NOT SUBMIT without a detailed description of generate_extra_data.
"""

from __future__ import absolute_import  # Not necessary in a Python 3-only module
from __future__ import division  # Not necessary in a Python 3-only module
from __future__ import google_type_annotations  # Not necessary in a Python 3-only module
from __future__ import print_function  # Not necessary in a Python 3-only module

from absl import app
from absl import flags

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
# Lint as: python3
"""Generate data for the listening tests to get saliency of subspectra of sound.

See README for full specification.
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
    "extra_data",
    "Where to save the dataset and the plots.")
flags.DEFINE_integer(
    "seed",
    1,
    "Seed used for libraries that use randomness.")
flags.DEFINE_integer("min_frequency", 20, "Minimum frequency for a tone.")
flags.DEFINE_integer("max_frequency", 20000, "Maximum frequency for a tone.")
flags.DEFINE_integer(
    "clip_db",
    80,
    "At what SPL to clip the dB of the generated examples.")
flags.DEFINE_integer(
    "sample_rate",
    48000,
    "Sample rate of audio that will be synthesized.")
flags.DEFINE_integer(
    "window_size",
    2048,
    "Window size of FFT that will be done on the audio.")
flags.DEFINE_integer(
    "cover_n_cbs_below_masker",
    4,
    "How many critical bands below the masker frequency should be exempt from "
    "`critical_bands_apart_masker` and should therefore all be covered as "
    "probes. ")
flags.DEFINE_integer(
    "cover_n_cbs_above_masker",
    8,
    "How many critical bands above the masker frequency should be exempt from "
    "`critical_bands_apart_masker` and should therefore all be covered as "
    "probes. ")
flags.DEFINE_integer(
    "min_db_masker",
    40,
    "Minimal dB SPL for the masker.")
flags.DEFINE_integer(
    "min_db",
    20,
    "Minimal dB SPL for the probe.")


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

  # TODO: define this elsewhere.
  critical_bands = [
      FLAGS.min_frequency, 100, 200, 300, 400, 505, 630, 770, 915, 1080, 1265,
      1475, 1720, 1990, 2310, 2690, 3125, 3675, 4350, 5250, 6350, 7650, 9400,
      11750, 15250, FLAGS.max_frequency
  ]

  # Generate the data.
  data = data_generation.generate_extra_data(
      critical_bands=critical_bands,
      sample_rate=FLAGS.sample_rate,
      window_size=FLAGS.window_size,
      cbs_below_masker=FLAGS.cover_n_cbs_below_masker,
      cbs_above_masker=FLAGS.cover_n_cbs_above_masker,
      minimum_db_masker=FLAGS.min_db_masker,
      minimum_db_probe=FLAGS.min_db,
      maximum_db=FLAGS.clip_db)

  # Run and save some analysis on the generated data.
  total_num_examples = data_analysis.save_two_tone_set(
      data, {}, critical_bands, save_directory, seed=FLAGS.seed)


if __name__ == "__main__":
  app.run(main)
