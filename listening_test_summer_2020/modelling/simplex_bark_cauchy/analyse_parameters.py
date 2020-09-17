#!/usr/bin/env python3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
"""Analyzes learned parameters on the data and plots all learned curves."""
import os
from typing import Tuple, List

from absl import app
from absl import flags
import dataset
from matplotlib import collections as matcoll
import matplotlib.pyplot as plt
import model
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string("logs_file", "all_vars_bark_cauchy_run_3.txt", "Where the logs are saved.")
flags.DEFINE_string("new_logs_file", "all_pars.txt",
                    "Where the logs are saved.")
flags.DEFINE_bool("plot_all_curves", True,
                  "Whether or not to plot all learned curves.")
flags.DEFINE_string("save_dir", "plots", "Where the output should be saved.")


def parse_logs(
    path_to_logs: str,
    path_to_pars: str) -> Tuple[List[float], List[float], float, float]:
  new_lines = []
  with open(path_to_logs, "r") as infile:
    mask_frequencies = []
    probe_levels = []
    mask_levels = []
    losses = []
    learned_vars = []
    for i, line in enumerate(infile):
      if i == 0:
        new_lines.append(line.strip("\n"))
        continue
      splitted_line = line.split(",")
      mask_frequencies.append(float(splitted_line[0]))
      probe_levels.append(float(splitted_line[1]))
      mask_levels.append(float(splitted_line[2]))
      losses.append(float(splitted_line[3]))
      model_class = model.Model(float(splitted_line[0]),
                                int(splitted_line[1]),
                                int(splitted_line[2]))
      actual_parameters = model_class.parameters_from_learned(
          [float(entry) for entry in splitted_line[4:]])
      learned_vars.append(actual_parameters)
      new_line = splitted_line[:4] + [
          " " + str(par) for par in actual_parameters
      ]
      new_lines.append(",".join(new_line))
  with open(path_to_pars, "w") as outfile:
    for line in new_lines:
      outfile.write(line)
      outfile.write("\n")
  return mask_frequencies, probe_levels, mask_levels, losses, learned_vars


def plot_curve(model_class, masking_frequency: float, probe_level: int,
               masking_level: int, loss: float, learned_vars: List[float],
               save_path: str, dataset, other_axis, all_losses=List[int],
               baseline_losses=List[int]) -> str:
  other_axis.set_title("Masker Frequency {} Hz, Probe Level {} dB".format(
        masking_frequency, probe_level), fontsize=12)
  masking_level_to_color = {40: "#1f77b4", 60: "#aec7e8", 80: "#ff7f0e"}

  # Get the data and model for this specs
  data = dataset.get_curve_data(
      masking_frequency=masking_frequency,
      probe_level=probe_level,
      masking_level=masking_level)
  actual_frequencies, actual_amplitudes = zip(*data)
  highest_y = max(actual_amplitudes) + 1
  all_losses.append(loss / len(actual_frequencies))

  # Do inference with the model
  predicted_amplitudes = []
  for frequency in actual_frequencies:
    current_inputs = [frequency
                     ] + learned_vars
    predicted_amplitudes.append(model_class.function(*current_inputs))

  # Sample 1000 points from the learned curve
  sampled_frequencies = np.linspace(
      min(actual_frequencies), max(actual_frequencies), 1000)
  sampled_frequencies.sort()
  sampled_amplitudes = []
  for freq in sampled_frequencies:
    current_inputs = [freq] + learned_vars
    amps = model_class.function(*current_inputs)
    sampled_amplitudes.append(amps)

  # Plot everything individually and on the grid
  error_lines = []
  baseline_errors = []
  for f, ampl_tuple in zip(actual_frequencies,
                           zip(actual_amplitudes, predicted_amplitudes)):
    pair = [(f, ampl_tuple[0]), (f, ampl_tuple[1])]
    error_lines.append(pair)
    baseline_errors.append((0 - ampl_tuple[0])**2)
  baseline_losses.append(sum(baseline_errors) / len(baseline_errors))
  linecoll = matcoll.LineCollection(error_lines, colors="k")
  linecoll2 = matcoll.LineCollection(error_lines, colors="k")
  # Plot fitted data
  cur_fig, ax = plt.subplots(num=2)
  other_axis.axvline(x=model_class.masker_frequency_bark, color="C2", ls=":")
  ax.axvline(model_class.masker_frequency_bark, c="c", label="Fixed Loc", ls=":")
  ax.scatter(actual_frequencies, actual_amplitudes, c="b", label="Actual")
  other_axis.scatter(
      actual_frequencies,
      actual_amplitudes,
      label="Actual, Masker Level {}".format(masking_level))
  ax.scatter(actual_frequencies, predicted_amplitudes, c="r", label="Predicted")
  ax.plot(sampled_frequencies, sampled_amplitudes)
  other_axis.plot(
      sampled_frequencies,
      sampled_amplitudes,
      label="Learned, Masker Level {}".format(masking_level))
  ax.add_collection(linecoll)
  ax.legend()
  other_axis.legend()
  other_axis.add_collection(linecoll2)
  plt.rcParams.update({"font.size": 8})
  title = "Nelder-Mead \n Current Loss: %.2f, " % (
      loss / len(actual_frequencies)) + model_class.parameter_repr(learned_vars)
  ax.set_ylim(0, highest_y)
  ax.set_xlim(0, 24)
  ax.set_title(title)
  other_axis.set_ylim(0, highest_y)
  other_axis.set_xlim(0, 24)
  filename = os.path.join(
      save_path, "masker_{}_probe_level_{}_masking_level_{}.png".format(
          masking_frequency, probe_level, masking_level))
  plt.savefig(filename, dpi=cur_fig.dpi)
  plt.close(cur_fig)
  return filename, other_axis


def plot_all_learned_curves(mask_frequencies: List[float],
                            probe_levels: List[int],
                            mask_levels: List[int],
                            losses: List[float],
                            learned_vars: List[List[float]],
                            save_dir: str):
  # Read training data.
  ds = dataset.MaskingDataset()
  for filename in os.listdir("data"):
    if filename.startswith("masker") and filename.endswith(".txt"):
      ds.read_data("data", filename)

  # Prepare gridplot of all learned curves.
  mask_frequencies_sorted = list(set(mask_frequencies.copy()))
  probe_levels_sorted = list(set(probe_levels.copy()))
  mask_frequencies_sorted.sort()
  probe_levels_sorted.sort()
  freq_to_y_axis = {freq: i for i, freq in enumerate(mask_frequencies_sorted)}
  probe_to_x_axis = {probe: i for i, probe in enumerate(probe_levels_sorted)}
  fig, axs = plt.subplots(len(mask_frequencies_sorted),
                          len(probe_levels_sorted), figsize=(14, 20), num=1)
  fig.text(
      0.5, 0.02, "Probe Frequency (bark)", ha="center", va="center", fontsize=14)
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
  for axes in axs:
    for ax in axes:
      ax.set_prop_cycle(color=[
          "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
          "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
          "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
          "#17becf", "#9edae5"
      ])
  all_losses = []
  baseline_losses = []
  for mask_frequency, probe_level, mask_level, loss, l_vars in zip(
      mask_frequencies, probe_levels, mask_levels, losses, learned_vars):
    model_class = model.Model(mask_frequency, probe_level, mask_level)
    # model_class.learned_pars = learned_vars
    other_axis = axs[freq_to_y_axis[mask_frequency]][
                     probe_to_x_axis[probe_level]]
    _, other_axis = plot_curve(model_class, mask_frequency, probe_level,
                               mask_level, loss, l_vars, save_dir, ds,
                               other_axis, all_losses, baseline_losses)
    axs[freq_to_y_axis[mask_frequency]][probe_to_x_axis[probe_level]] = other_axis
  print("Baseline loss per curve: {}".format(sum(baseline_losses) / len(baseline_losses)))
  print("Mean loss per curve: {}".format(sum(all_losses) / len(all_losses)))
  save_filename = os.path.join(save_dir,
                               "all_learned_masking_patterns.png")
  plt.savefig(
      save_filename,
      dpi=fig.dpi)
  return save_filename


def analyze_pars_per_level_diff(mask_frequencies: List[float],
                                probe_levels: List[int], mask_levels: List[int],
                                learned_vars: List[float], save_path: str,
                                par_name: str) -> str:
  all_pars = {
      m - p: {
          "x": [],
          "y": []
      } for m in set(mask_levels) for p in set(probe_levels)
  }
  _, ax = plt.subplots(figsize=(12, 16))
  mask_probe_diff = list(all_pars.keys())
  mask_probe_diff.sort()
  for mask_freq, probe_level, mask_level, l_vars in zip(mask_frequencies,
                                                        probe_levels,
                                                        mask_levels,
                                                        learned_vars):
    all_pars[mask_level - probe_level]["x"].append(mask_freq)
    all_pars[mask_level - probe_level]["y"].append(l_vars)

  ax.set_prop_cycle(color=[
      "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
      "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
      "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
      "#17becf", "#9edae5"
  ])
  ax.set_title("{} against mask frequencies.".format(par_name))
  ax.set_xlabel("Masker Frequency (Hz)", fontsize=12)
  ax.set_ylabel(par_name, fontsize=12)
  for probe_masker_difference in mask_probe_diff:
    plots = all_pars[probe_masker_difference]
    ax.scatter(plots["x"], plots["y"],
               label="Masker Probe level difference {}".format(
                   probe_masker_difference))
  ax.legend()
  save_filename = os.path.join(save_path, "mask_frequencies_{}.png".format(par_name))
  plt.savefig(save_filename)
  return save_filename


def analyze_pars_per_masker(mask_frequencies: List[float],
                            mask_levels: List[int],
                            probe_levels: List[int],
                            learned_vars: List[float],
                            par_name: str,
                            save_path: str) -> str:
  _, ax = plt.subplots(figsize=(12, 16))
  ax.set_xlabel("Masker Frequency (Hz)", fontsize=12)
  ax.set_ylabel(par_name, fontsize=12)
  all_pars = {mask_freq: {"x": [], "y": []} for mask_freq in set(mask_frequencies)}
  mask_freqs_sorted = list(all_pars.keys())
  mask_freqs_sorted.sort()
  x = [m - p for m, p in zip(mask_levels, probe_levels)]
  for mask_freq, mask_probe_diff, l_var in zip(mask_frequencies,
                                                x, learned_vars):
    all_pars[mask_freq]["x"].append(mask_probe_diff)
    all_pars[mask_freq]["y"].append(l_var)
  ax.set_prop_cycle(color=[
      "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
      "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
      "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
      "#17becf", "#9edae5"
  ])
  ax.set_title("{} against masker level - probe level.".format(par_name))
  ax.set_xlabel("masker level - probe level (dB)", fontsize=12)
  ax.set_ylabel(par_name, fontsize=12)
  for mask_freq in mask_freqs_sorted:
    plots = all_pars[mask_freq]
    ax.scatter(plots["x"], plots["y"],
               label="Mask Freq {}".format(
                   mask_freq))
  ax.legend()
  save_filename = os.path.join(save_path, "mask_frequencies_{}.png".format(par_name))
  plt.savefig(save_filename)
  return save_filename


def main(argv):

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not os.path.exists(FLAGS.save_dir):
    os.mkdir(FLAGS.save_dir)

  (mask_frequencies, probe_levels, mask_levels, losses,
   learned_vars) = parse_logs(FLAGS.logs_file, FLAGS.new_logs_file)
  # mask_frequencies = [m / 1000 for m, l in zip(mask_frequencies, losses) if l <= 1]
  # probe_levels = [p / 10 for p, l in zip(probe_levels, losses) if l <= 1]
  # mask_levels = [m / 10 for m, l in zip(mask_levels, losses) if l <= 1]
  if FLAGS.plot_all_curves:
    plot_all_learned_curves(mask_frequencies,
                            probe_levels,
                            mask_levels,
                            losses,
                            learned_vars,
                            FLAGS.save_dir)

  analyze_pars_per_level_diff(
      mask_frequencies,
      probe_levels,
      mask_levels, [v[1] for v in learned_vars],
      FLAGS.save_dir,
      par_name="Alpha")
  analyze_pars_per_masker(mask_frequencies, probe_levels, mask_levels,
                          [v[0] for v in learned_vars], "Amplitude",
                          FLAGS.save_dir)
  analyze_pars_per_masker(mask_frequencies, probe_levels, mask_levels,
                          [v[2] for v in learned_vars], "Scale",
                          FLAGS.save_dir)

if __name__ == "__main__":
  app.run(main)
