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
import dataset_threetone
import os
import numpy as np
from matplotlib import collections as matcoll
import matplotlib.pyplot as plt
from absl import app
from absl import flags

FLAGS = flags.FLAGS


flags.DEFINE_string("vars_file", "all_vars_bark_sk_run_3.txt",
                    "Where the logs are saved.")
flags.DEFINE_string("data_path", "ood_test_threetone",
                    "Where the test data is .")
flags.DEFINE_bool("plot_all_curves", True,
                  "Whether or not to plot all learned curves.")
flags.DEFINE_bool("plot_ood_curves", False,
                  "Whether or not to plot OOD curves.")
flags.DEFINE_string("save_dir", "predicted_plots", "Where the output should be saved.")



def plot_curve(model_class, masking_frequency: float, probe_level: int, masker_frequency_bark: int,
               masking_level: int,
               learned_pars: List[float], save_path: str, dataset,
               other_axis=None) -> str:
  if other_axis:
    other_axis.set_title("Masker Frequency {} Hz, Probe Level {} dB".format(
          masking_frequency, probe_level), fontsize=12)
  masking_level_to_color = {40: "#1f77b4", 60: "#aec7e8", 80: "#ff7f0e"}

  # Get the data and model for this specs
  try:
    data = dataset.get_curve_data(
      masking_frequency=masking_frequency,
      probe_level=probe_level,
      masking_level=masking_level)
  except ValueError:
    return "", None
  actual_frequencies, actual_amplitudes = zip(*data)
  highest_y = max(actual_amplitudes) + 1

  # Do inference with the model
  predicted_amplitudes = []
  for frequency in actual_frequencies:
    prediction = model_class.predict(masking_frequency, probe_level,
                                     masking_level, frequency)
    predicted_amplitudes.append(prediction)

  # Sample 1000 points from the learned curve
  sampled_frequencies = list(range(0, 25))
  sampled_frequencies.sort()
  sampled_amplitudes = []
  for freq in sampled_frequencies:
    prediction = model_class.predict(masking_frequency, probe_level,
                                     masking_level, freq)
    sampled_amplitudes.append(prediction)

  # Plot everything individually and on the grid
  error_lines = []
  for f, ampl_tuple in zip(actual_frequencies,
                           zip(actual_amplitudes, predicted_amplitudes)):
    pair = [(f, ampl_tuple[0]), (f, ampl_tuple[1])]
    error_lines.append(pair)
  linecoll = matcoll.LineCollection(error_lines, colors="k")
  linecoll2 = matcoll.LineCollection(error_lines, colors="k")
  # Plot fitted data
  cur_fig, ax = plt.subplots(num=2)
  if other_axis:
    other_axis.axvline(x=masker_frequency_bark, color="C2", ls=":")
    other_axis.scatter(
      actual_frequencies,
      actual_amplitudes,
      label="Actual, Masker Level {}".format(masking_level))
    other_axis.plot(
      sampled_frequencies,
      sampled_amplitudes,
      label="Learned, Masker Level {}".format(masking_level), ls="--")
    other_axis.legend()
    other_axis.add_collection(linecoll2)
    other_axis.set_ylim(0, highest_y)
    other_axis.set_xlim(0, 24)
  ax.axvline(masker_frequency_bark, c="c", label="Fixed Loc", ls=":")
  ax.scatter(actual_frequencies, actual_amplitudes, c="b", label="Actual")

  ax.scatter(actual_frequencies, predicted_amplitudes, c="r", label="Predicted",
             ls="--")
  ax.plot(sampled_frequencies, sampled_amplitudes)

  ax.add_collection(linecoll)
  ax.legend()

  plt.rcParams.update({"font.size": 8})
  title = "Predictions on Test Set"
  ax.set_ylim(0, highest_y)
  ax.set_xlim(0, 24)
  ax.set_title(title)
  filename = os.path.join(
      save_path, "masker_{}_probe_level_{}_masking_level_{}.png".format(
          masking_frequency, probe_level, masking_level))
  print(filename)
  plt.savefig(filename, dpi=cur_fig.dpi)
  plt.close(cur_fig)
  return filename, other_axis


def plot_all_learned_curves(ds,
                            mask_frequencies: str,
                            probe_level: int,
                            mask_level: int,
                            model_class: model.FullModel,
                            save_dir: str):

  # Prepare gridplot of all learned curves.
  fig, ax = plt.subplots(1, 1, figsize=(8, 7))
  fig.tight_layout(pad=6.0)
  ax.set_prop_cycle(color=[
      "#ff7f0e", "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
      "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
      "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
      "#17becf", "#9edae5"
  ])
  fig.text(
      0.5, 0.02, "Probe Frequency (Bark)", ha="center", va="center", fontsize=14)
  fig.text(
      0.02,
      0.5,
      "Masked SPL (B)",
      ha="center",
      va="center",
      rotation="vertical",
      fontsize=14)
  # ax.set_xlabel("Probe Frequency (Bark)", fontsize=12)
  # ax.set_ylabel("Masked SPL (dB)", fontsize=12)
  ax.set_title("Masker Frequencies {} Hz, Probe Level {} dB".format(
      mask_frequencies, probe_level), fontsize=14)
  mask_freq_one, mask_freq_two = mask_frequencies.split("+")
  mask_freq_one = float(mask_freq_one)
  mask_freq_two = float(mask_freq_two)
  plt.axvline(x=float(mask_freq_one), color="C2", ls=":")
  plt.axvline(x=float(mask_freq_two), color="C2", ls=":")
  plt.axhline(y=probe_level, color="C2", ls=":")

  # Get the data and model for this specs
  try:
    data = ds.get_curve_data(
      masking_frequency=mask_frequencies,
      probe_level=probe_level,
      masking_level=mask_level)
  except ValueError:
    return "", None

  actual_frequencies, actual_amplitudes = zip(*data)
  highest_y = max(actual_amplitudes) + 1

  # Do inference with the model
  predicted_amplitudes = []
  for frequency in actual_frequencies:
    prediction = model_class.predict(mask_frequencies, probe_level,
                                     mask_level, frequency)
    predicted_amplitudes.append(prediction)

  # Sample 1000 points from the learned curve
  sampled_frequencies = list(range(0, 25))
  sampled_frequencies.sort()
  sampled_amplitudes = []
  for freq in sampled_frequencies:
    prediction = model_class.predict(mask_frequencies, probe_level,
                                     mask_level, freq)
    sampled_amplitudes.append(prediction)

  # Plot everything individually and on the grid
  error_lines = []
  for f, ampl_tuple in zip(actual_frequencies,
                           zip(actual_amplitudes, predicted_amplitudes)):
    pair = [(f, ampl_tuple[0]), (f, ampl_tuple[1])]
    error_lines.append(pair)
  linecoll2 = matcoll.LineCollection(error_lines, colors="k")
  # Plot fitted data
  masker_frequency_bark_one = ds.frequency_to_cb(mask_freq_one)
  masker_frequency_bark_two = ds.frequency_to_cb(mask_freq_two)
  ax.axvline(x=masker_frequency_bark_one, color="C2", ls=":")
  ax.axvline(x=masker_frequency_bark_two, color="C2", ls=":")
  ax.scatter(
      actual_frequencies,
      actual_amplitudes,
      label="Actual, Masker Level {}".format(mask_level))
  ax.plot(
      sampled_frequencies,
      sampled_amplitudes,
      label="Learned, Masker Level {}".format(mask_level),
      ls="--")
  ax.legend()
  ax.add_collection(linecoll2)
  ax.set_ylim(0, highest_y)
  ax.set_xlim(0, 24)

  plt.rcParams.update({"font.size": 8})
  title = "Predictions on Test Set"
  filename = os.path.join(
      save_dir, "maskers_{}_probe_level_{}_masking_level_{}.png".format(
          mask_frequencies, probe_level, mask_level))
  plt.savefig(filename)
  return filename


def plot_ood_curves(ds, model_class, masker_frequencies, probe_level, masker_level,
                    save_dir):
  _, ax = plt.subplots(1, 1, figsize=(6, 6))
  ax.set_prop_cycle(color=[
      "#ff7f0e", "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
      "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
      "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
      "#17becf", "#9edae5"
  ])
  ax.set_xlabel("Probe Frequency (Bark)", fontsize=12)
  ax.set_ylabel("Masked SPL (dB)", fontsize=12)
  ax.set_title("Masker Frequencies {} Hz, Probe Level {} dB".format(
      masker_frequencies, probe_level), fontsize=14)
  mask_freq_one, mask_freq_two = masker_frequencies.split("+")
  plt.axvline(x=float(mask_freq_one), color="C2", ls=":")
  plt.axvline(x=float(mask_freq_two), color="C2", ls=":")
  plt.axhline(y=probe_level, color="C2", ls=":")
  current_levels = []
  mean_masking_per_level = []
  current_levels.append(masker_level)
  for example in ds.get_all_data():
    prediction = model_class.predict(example["masker_frequency"],
                                     example["probe_level"],
                                     example["masker_level"],
                                     example["probe_frequency_bark"])
    error = (example["target_bel_masking"] - prediction)**2
  frequencies = masker_curve["probe_frequencies"]
  masking = masker_curve["probe_masking"]
  plt.errorbar(
      frequencies,
      average_masking,
      yerr=std_masking,
      fmt="o",
      label="Masker Level {}".format(masker_level))
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels)
  save_file = "masker_{}_probe_{}".format(masker_frequencies, probe_level)
  with open(os.path.join(save_directory, save_file + ".txt"), "wt") as infile:
    for level, mean_masking in zip(current_levels, mean_masking_per_level):
      infile.write("level {}: ".format(level))
      freq_maskers_str = ";".join(["{},{}".format(f, m) for f, m in zip(
          frequencies, mean_masking)])
      infile.write(freq_maskers_str)
      infile.write("\n")
  plt.savefig(os.path.join(save_directory,
                               save_file + ".png"))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not os.path.exists(FLAGS.vars_file):
    raise ValueError("No data found at %s" % FLAGS.vars_file)

  if not os.path.exists(FLAGS.save_dir):
    os.mkdir(FLAGS.save_dir)

  ds = dataset_threetone.MaskingDataset()
  for filename in os.listdir(FLAGS.data_path):
    if filename.startswith("masker") and filename.endswith(".txt"):
      ds.read_data(FLAGS.data_path, filename)

  model_class = model.FullModel()
  model_class.initialize_models(FLAGS.vars_file)
  total_error = 0
  total_points = 0
  baseline_error = 0
  for example in ds.get_all_data():
    prediction = model_class.predict(example["masker_frequency"],
                                     example["probe_level"],
                                     example["masker_level"],
                                     example["probe_frequency_bark"])
    error = (example["target_bel_masking"] - prediction)**2
    baseline_error += (example["target_bel_masking"] - 0)**2
    total_error += error
    total_points += 1
  print("Baseline Mean squared error: {}".format(baseline_error / total_points))
  print("Mean squared error: {}".format(total_error / total_points))
  if FLAGS.plot_all_curves:
    plot_all_learned_curves(ds,
                            "1370+2908",
                            60,
                            80,
                            model_class,
                            FLAGS.save_dir)
  if FLAGS.plot_ood_curves:
    mask_frequencies = [1370.0]
    probe_levels = [30]
    mask_levels = [80]
    plot_all_learned_curves(ds, "1370.0+2908.0", probe_levels, mask_levels,
                            model_class, FLAGS.save_dir)


if __name__ == "__main__":
  app.run(main)