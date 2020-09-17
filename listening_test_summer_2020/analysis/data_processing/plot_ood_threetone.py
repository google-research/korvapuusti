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
import os
import numpy as np
from matplotlib import collections as matcoll
import matplotlib.pyplot as plt
from absl import app
from absl import flags

FLAGS = flags.FLAGS


flags.DEFINE_string("data_path", "ood_threetone_threetone",
                    "Where the test data is.")
flags.DEFINE_string("input_file_name", "ood_threetone_preds.txt",
                    "Where the test data is .")
flags.DEFINE_string("save_dir", "predicted_plots", "Where the output should be saved.")


def binary_search(arr, item):
  """Finds closest index to the left of an item in arr."""
  low = 0
  high = len(arr) - 1
  mid = 0

  while low <= high:
    mid = (high + low) // 2
    # Check if item is present at mid
    if arr[mid] < item:
      low = mid
    # If item is greater, ignore left half
    elif arr[mid] > item:
      high = mid
    # If item is smaller, ignore right half
    else:
      return mid
    if arr[high] <= item:
      return high

    if arr[low] <= item < arr[low + 1]:
      return low
  return mid


def frequency_to_cb(frequency: float, critical_bands: List[int],
                    sample_rate: int, window_size: int) -> int:
  cb = binary_search(critical_bands, frequency)
  return cb


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
                            save_dir: str,
                            critical_bands: List[int]):

  # Prepare gridplot of all learned curves.
  fig, ax = plt.subplots(1, 1, figsize=(8, 7))
  fig.tight_layout(pad=6.0)
  ax.set_prop_cycle(color=[
      "#ff7f0e", "#ff7f0e", "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
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
  actual_frequencies, actual_amplitudes, predicted_amplitudes = [], [], []
  for example in ds:
    probe_frequency = example["probe_frequency_bark"]
    amplitude = example["target_bel_masking"]
    actual_frequencies.append(probe_frequency)
    actual_amplitudes.append(amplitude)
    predicted_amplitudes.append(example["predicted_bel_masking"])
    print(amplitude, example["predicted_bel_masking"])

  highest_y = max(actual_amplitudes) + 1


#   # Sample 1000 points from the learned curve
#   sampled_frequencies = list(range(0, 25))
#   sampled_frequencies.sort()
#   sampled_amplitudes = []
#   for freq in sampled_frequencies:
#     prediction = model_class.predict(mask_frequencies, probe_level,
#                                      mask_level, freq)
#     sampled_amplitudes.append(prediction)

  # Plot everything individually and on the grid
  error_lines = []
  for f, ampl_tuple in zip(actual_frequencies,
                           zip(actual_amplitudes, predicted_amplitudes)):
    pair = [(f, ampl_tuple[0]), (f, ampl_tuple[1])]
    error_lines.append(pair)
  linecoll2 = matcoll.LineCollection(error_lines, colors="k")
  # Plot fitted data
  print(mask_freq_one, mask_freq_two)
  masker_frequency_bark_one = frequency_to_cb(1370.0,
                                              critical_bands,
                                              48000,
                                              2024)
  masker_frequency_bark_two = frequency_to_cb(2908.0,
                                              critical_bands,
                                              48000,
                                              2024)
  ax.axvline(x=masker_frequency_bark_one, color="C2", ls=":")
  ax.axvline(x=masker_frequency_bark_two, color="C2", ls=":")
  ax.scatter(
      actual_frequencies,
      actual_amplitudes,
      label="Actual, Masker Level {}".format(mask_level))
  # ax.plot(
  #     sampled_frequencies,
  #     sampled_amplitudes,
  #     label="Learned, Masker Level {}".format(mask_level),
  #     ls="--")
  ax.scatter(
        actual_frequencies,
        predicted_amplitudes,
        label="Predicted",
        marker="x")
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


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not os.path.exists(FLAGS.save_dir):
    os.mkdir(FLAGS.save_dir)

  critical_bands = [
      20, 100, 200, 300, 400, 505, 630, 770, 915, 1080, 1265, 1475, 1720, 1990,
      2310, 2690, 3125, 3675, 4350, 5250, 6350, 7650, 9400, 11750, 15250, 20000
  ]

  ds = []
  total_mse = 0
  total_points = 0
  skipped_lines = 0
  with open(os.path.join(FLAGS.data_path,
                         FLAGS.input_file_name), "r") as infile:
    lines = infile.readlines()
    for i, line in enumerate(lines):
      if i == 0:
        continue
      splitted_line = line.split(";")
      if len(splitted_line) != 7:
        skipped_lines += 1
        continue
      frequencies = splitted_line[0].split(",")
      masker_frequency = float(frequencies[0])
      masker_frequency_one = 1370.0
      masker_frequency_two = 2908.0
      probe_frequency = float(frequencies[1])
      levels = splitted_line[1].split(",")
      masker_level = float(levels[0])
      probe_level = float(levels[1])
      target_bel_masking = float(splitted_line[5]) / 10
      predicted_bel_masking = float(splitted_line[6]) / 10
      total_mse += ((target_bel_masking) - (predicted_bel_masking))**2
      total_points += 1
      probe_frequency_bark = frequency_to_cb(probe_frequency,
                                             critical_bands,
                                             48000,
                                             2024)
      masker_frequency_bark_one = frequency_to_cb(masker_frequency_one,
                                              critical_bands,
                                              48000,
                                              2024)
      masker_frequency_bark_two = frequency_to_cb(masker_frequency_two,
                                              critical_bands,
                                              48000,
                                              2024)
      ds.append({
          "probe_frequency": probe_frequency,
          "probe_frequency_bark": probe_frequency_bark,
          "masker_frequency": masker_frequency,
          "masker_frequency_bark_one": masker_frequency_bark_one,
          "masker_frequency_bark_two": masker_frequency_bark_two,
          "probe_level": probe_level,
          "masker_level": masker_level,
          "target_bel_masking": target_bel_masking,
          "predicted_bel_masking": predicted_bel_masking
      })

  plot_all_learned_curves(ds,
                          "1370+2908",
                          60,
                          80,
                          FLAGS.save_dir,
                          critical_bands)
  print("Recalculated MSE: ", total_mse / total_points)
  print("For %d points." % total_points)
  assert skipped_lines == 3, "Skipped too few or many lines."


if __name__ == "__main__":
  app.run(main)
