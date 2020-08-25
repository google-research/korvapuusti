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
from absl import flags
import json

FLAGS = flags.FLAGS


flags.DEFINE_string("vars_file", "all_vars_bark_sk_run_2.txt",
                    "Where the logs are saved.")
flags.DEFINE_string("data_path", "ood_test_sine",
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
  plt.savefig(filename, dpi=cur_fig.dpi)
  plt.close(cur_fig)
  return filename, other_axis


def plot_all_learned_curves(ds,
                            mask_frequencies: List[float],
                            probe_levels: List[int],
                            mask_levels: List[int],
                            model_class: model.FullModel,
                            save_dir: str):

  # Prepare gridplot of all learned curves.
  mask_frequencies_sorted = list(set(mask_frequencies.copy()))
  probe_levels_sorted = list(set(probe_levels.copy()))
  mask_frequencies_sorted.sort()
  probe_levels_sorted.sort()
  freq_to_y_axis = {freq: i for i, freq in enumerate(mask_frequencies_sorted)}
  probe_to_x_axis = {probe: i for i, probe in enumerate(probe_levels_sorted)}
  fig, axs = plt.subplots(len(mask_frequencies_sorted),
                          len(probe_levels_sorted), figsize=(16, 7), num=1)
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
  fig.tight_layout(pad=6.0)
  if len(mask_frequencies_sorted) == 1 and len(probe_levels_sorted) == 1:
    axs = [[axs]]
  elif len(mask_frequencies_sorted) == 1 or len(probe_levels_sorted) == 1:
    axs = [axs]
  for axes in axs:
    for ax in axes:
      ax.set_prop_cycle(color=[
          "#ffbb78", "#2ca02c", "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
          "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
          "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
          "#17becf", "#9edae5"
      ])
  for mask_frequency, probe_level, mask_level in zip(
      mask_frequencies, probe_levels, mask_levels):
    selected_model = model_class.find_closest_model(mask_frequency, probe_level, mask_level)
    learned_pars = selected_model.learned_pars
    # model_class.learned_pars = learned_vars
    other_axis = axs[freq_to_y_axis[mask_frequency]][
                     probe_to_x_axis[probe_level]]
    _, other_axis = plot_curve(model_class, mask_frequency, probe_level,
                               selected_model.masker_frequency_bark,
                               mask_level, learned_pars, save_dir, ds,
                               other_axis)
    if other_axis:
      axs[freq_to_y_axis[mask_frequency]][probe_to_x_axis[probe_level]] = other_axis
  save_filename = os.path.join(save_dir,
                               "all_predictions.png")
  plt.savefig(
      save_filename,
      dpi=fig.dpi)
  return save_filename


def plot_ood_curves(ds, model_class, mask_frequencies, probe_levels, mask_levels,
                    save_dir):

  for mask_frequency, probe_level, mask_level in zip(
      mask_frequencies, probe_levels, mask_levels):
    learned_pars = model_class.find_closest_model(mask_frequency, probe_level,
                                                  mask_level).learned_pars
    # model_class.learned_pars = learned_vars
    plot_curve(model_class, mask_frequency, probe_level, mask_level,
               learned_pars, save_dir, ds)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not os.path.exists(FLAGS.vars_file):
    raise ValueError("No data found at %s" % FLAGS.vars_file)

  if not os.path.exists(FLAGS.save_dir):
    os.mkdir(FLAGS.save_dir)

  ds = dataset.MaskingDataset()
  for filename in os.listdir(FLAGS.data_path):
    if filename.startswith("masker") and filename.endswith(".txt"):
      ds.read_data(FLAGS.data_path, filename)


  ds_two = []
  with open("predictions.txt", "r") as infile:
    lines = infile.readlines()
    for i, line in enumerate(lines):
      if i == 0:
        continue
      splitted_line = line.split(";")
      frequencies = splitted_line[0].split(",")
      masker_frequency = float(frequencies[0])
      probe_frequency = float(frequencies[1])
      levels = splitted_line[1].split(",")
      masker_level = float(levels[0])
      probe_level = float(levels[1])
      target_bel_masking = float(splitted_line[5]) / 10
      ds_two.append({
          "probe_frequency": probe_frequency,
          "masker_frequency": masker_frequency,
          "probe_level": probe_level,
          "masker_level": masker_level,
          "target_bel_masking": target_bel_masking
      })

  model_class = model.FullModel()
  model_class.initialize_models(FLAGS.vars_file)
  total_error = 0
  total_points = 0
  baseline_error = 0
  for example in ds.get_all_data():
    # bark_frequency = ds.frequency_to_cb(example["probe_frequency"])
    actual_level = example["probe_level"]
    target_bel_masking = example["target_bel_masking"]
    prediction = model_class.predict(example["masker_frequency"],
                                     example["probe_level"],
                                     example["masker_level"],
                                     example["probe_frequency_bark"])
    error = (target_bel_masking - prediction)**2
    baseline_error += (target_bel_masking - 0)**2
    total_error += error
    total_points += 1
  print("Num predicted: ", total_points)
  print("Baseline Mean squared error: {}".format(baseline_error / total_points))
  print("Mean squared error: {}".format(total_error / total_points))
  if FLAGS.plot_all_curves:
    plot_all_learned_curves(ds,
                            [843.0, 843.0, 843.0, 843.0],
                            [20, 20, 40, 40],
                            [50, 70, 50, 70],
                            model_class,
                            FLAGS.save_dir)
  if FLAGS.plot_ood_curves:
    mask_frequencies = [1370.0]
    probe_levels = [30]
    mask_levels = [80]
    plot_all_learned_curves(ds, mask_frequencies, probe_levels, mask_levels,
                            model_class, FLAGS.save_dir)


if __name__ == "__main__":
  app.run(main)
