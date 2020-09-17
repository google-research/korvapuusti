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
"""Visualize features for TF model."""

import json
import os
import random
from typing import Dict, List, Any, Tuple
import data_plotting
import data_helpers
import pprint
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from absl import app
from absl import flags
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file_path",
    "tfdata/extratraindataraw_0000000-0000020.json",
    "JSON file with answers per annotator.")
flags.DEFINE_string(
    "output_directory",
    "output",
    "Directory to save preprocessed data in.")


def frequency_to_bin(frequency: float, sample_rate: int,
                     window_size: int) -> int:
  return round(frequency / (sample_rate / window_size))


def plot_fft(example: Dict[str, Any], save_directory: str, i: int):
  bins = example["bins"]
  frequency_bins = np.linspace(0.0, bins, bins)
  coefficient = example["coefficients"]

  fig, ax = plt.subplots()
  ax.set_title("(2) Fast Fourier Transform: Frequency Bin vs. Intensity")
  sample_rate = example["samplerate"]
  window_size = bins * 2
  frequency_bins_str = ", ".join([str(frequency_to_bin(freq, sample_rate, window_size)) for freq in example["frequencies"]])
  # ax.set_title("Frequencies: " + ", ".join([str(freq) for freq in example["frequencies"]]) + "\nBins: " + frequency_bins_str)
  ax.plot(frequency_bins, coefficient)
  ax.set_xticks(np.arange(bins, step=100))
  ax.set_xlabel("Frequency Bins")
  ax.set_ylabel("Intensity")
  spls_string = ", ".join([str(spl) for spl in example["spls"]])
  plt.savefig(os.path.join(save_directory, "fft{}.png".format(
      "_frequencies_" + ", ".join([str(freq) for freq in example["frequencies"]]) + "_bins_" + frequency_bins_str + "_spls_" + spls_string)))
  return


def plot_sine(example: Dict[str, Any], save_directory: str, i: int):
  timesteps = int(len(example["buffer"]) / 6)
  timesteps_x = np.linspace(0.0, timesteps, timesteps)
  buffer = example["buffer"][:timesteps]
  fig, ax = plt.subplots()
  sample_rate = example["samplerate"]
  window_size = example["bins"] * 2
  frequency_bins_str = ", ".join([str(frequency_to_bin(freq, sample_rate, window_size)) for freq in example["frequencies"]])
  ax.set_title("(1) Raw Audio: Time vs. Intensity")
  ax.plot(timesteps_x, buffer)
  ax.set_xticks(np.arange(timesteps, step=300))
  ax.set_xlabel("Time")
  ax.set_ylabel("Intensity")
  spls_string = ", ".join([str(spl) for spl in example["spls"]])
  plt.savefig(os.path.join(save_directory, "sine{}.png".format(
      "_frequencies_" + ", ".join([str(freq) for freq in example["frequencies"]]) + "_bins_" + frequency_bins_str + "_spls_" + spls_string)))
  return


def plot_snr(example: Dict[str, Any], save_directory: str, i: int):
  bins = example["bins"]
  frequency_bins = np.linspace(0.0, bins, bins)
  signal = np.array(example["spectrumsignal"])
  noise = np.array(example["spectrumnoise"])
  snr = signal / (signal + noise)

  fig, ax = plt.subplots()
  sample_rate = example["samplerate"]
  ax.set_title("(3) SNR: Frequency Bin vs. SNR")
  window_size = bins * 2
  frequency_bins_str = ", ".join([str(frequency_to_bin(freq, sample_rate, window_size)) for freq in example["frequencies"]])
  # ax.set_title("Frequencies: " + ", ".join([str(freq) for freq in example["frequencies"]]) + "\nBins: " + frequency_bins_str)
  ax.plot(frequency_bins, snr)
  ax.set_xticks(np.arange(bins, step=100))
  ax.set_xlabel("Frequency Bins")
  ax.set_ylabel("SNR")
  spls_string = ", ".join([str(spl) for spl in example["spls"]])
  plt.savefig(os.path.join(save_directory, "snr{}.png".format(
      "_frequencies_" + ", ".join([str(freq) for freq in example["frequencies"]]) + "_bins_" + frequency_bins_str + "_spls_" + spls_string)))
  return


def plot_carfac(example: Dict[str, Any], save_directory: str, i: int):
  bins = example["bins"]
  channels = example["channels"]
  frequency_bins = np.linspace(0.0, bins, bins)
  signal = np.array(example["spectrumsignalcarfac"])
  noise = np.array(example["spectrumnoisecarfac"])
  signal = signal.reshape((channels, bins))
  noise = noise.reshape((channels, bins))
  y, x = np.mgrid[slice(1, channels + 1, 1),
                  slice(1, bins + 1, 1)]

  fig, (ax0, ax1) = plt.subplots(nrows=2)
  sample_rate = example["samplerate"]
  window_size = bins * 2
  frequency_bins_str = ", ".join([str(frequency_to_bin(freq, sample_rate, window_size)) for freq in example["frequencies"]])
  ax0.set_title("(4) CAR-FAC\nSignal")
  im = ax0.pcolormesh(x, y, signal)
  ax0.set_xticks(np.arange(bins, step=100))
  ax0.set_yticks(np.arange(channels, step=10))
  ax0.set_ylabel("CAR-FAC Channels")
  fig.colorbar(im, ax=ax0)
  ax1.set_title("Noise")
  cf = ax1.pcolormesh(x, y, noise)
  fig.colorbar(cf, ax=ax1)
  ax1.set_xticks(np.arange(bins, step=100))
  ax1.set_yticks(np.arange(channels, step=10))
  ax1.set_xlabel("Frequency Bins")
  ax1.set_ylabel("CAR-FAC Channels")
  fig.tight_layout()
  spls_string = ", ".join([str(spl) for spl in example["spls"]])
  plt.savefig(os.path.join(save_directory, "carfac{}.png".format(
      "_frequencies_" + ", ".join([str(freq) for freq in example["frequencies"]]) + "_bins_" + frequency_bins_str + "_spls_" + spls_string)))
  return


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not os.path.exists(FLAGS.input_file_path):
    raise ValueError("No data found at %s" % FLAGS.input_file_path)

  if not os.path.exists(FLAGS.output_directory):
    os.mkdir(FLAGS.output_directory)

  with open(FLAGS.input_file_path, "r") as infile:
    data = json.load(infile)

  for i in range(5):
    plot_fft(data[i], FLAGS.output_directory, i)
    plot_sine(data[i], FLAGS.output_directory, i)
    plot_snr(data[i], FLAGS.output_directory, i)
    plot_carfac(data[i], FLAGS.output_directory, i)


if __name__ == "__main__":
  app.run(main)
