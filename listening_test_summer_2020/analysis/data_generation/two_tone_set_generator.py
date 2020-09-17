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
    "listening_data",
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
    "num_examples_per_loudness_curve",
    8,
    "How many listening tests to generate for each ISO equal loudness curve.")
flags.DEFINE_integer(
    "num_curves",
    3,
    "How many equal loudness curves (i.e., phons level) to use.")
flags.DEFINE_integer(
    "critical_bands_apart_probe",
    2,
    "How many critical bands should each frequency in a two-tone example be "
    "apart.")
flags.DEFINE_integer(
    "critical_bands_apart_masker",
    5,
    "How many critical bands should each frequency in a two-tone example be "
    "apart.")
flags.DEFINE_integer(
    "cover_n_cbs_below_masker",
    8,
    "How many critical bands below the masker frequency should be exempt from "
    "`critical_bands_apart_masker` and should therefore all be covered as "
    "probes. ")
flags.DEFINE_integer(
    "cover_n_cbs_above_masker",
    4,
    "How many critical bands above the masker frequency should be exempt from "
    "`critical_bands_apart_masker` and should therefore all be covered as "
    "probes. ")
flags.DEFINE_string(
    "probe_levels",
    "30,60",
    "Comma-separated string of SPLs to generate examples from.")
flags.DEFINE_string(
    "masker_levels",
    "40,60,80",
    "Comma-separated string of SPLs to generate examples from.")


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
  probe_levels = FLAGS.probe_levels.split(",")
  masker_levels = FLAGS.masker_levels.split(",")

  # Generate ISO data.
  iso_data = data_generation.generate_iso_repro_examples(
      num_examples_per_curve=FLAGS.num_examples_per_loudness_curve,
      num_curves=FLAGS.num_curves,
      clip_db=FLAGS.clip_db)

  # Plot the ISO data.
  _ = data_analysis.plot_iso_examples(iso_data, save_directory)

  # Generate the data.
  data = data_generation.generate_two_tone_set(
      critical_bands=critical_bands,
      probe_levels=probe_levels,
      masker_levels=masker_levels,
      critical_bands_apart_probe=FLAGS.critical_bands_apart_probe,
      critical_bands_apart_masker=FLAGS.critical_bands_apart_masker,
      all_lower_probes=FLAGS.cover_n_cbs_below_masker,
      all_higher_probes=FLAGS.cover_n_cbs_above_masker)

  # Run and save some analysis on the generated data.
  total_num_examples = data_analysis.save_two_tone_set(
      data, iso_data, critical_bands, save_directory, seed=FLAGS.seed)


if __name__ == "__main__":
  app.run(main)
