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
"""Plot data from listening tests."""

import os
from typing import List, Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_histogram(values: List[Any], path: str, bins=None, logscale=False,
                   x_label="", title=""):
  """Plots and saves a histogram."""
  _, ax = plt.subplots(1, 1)
  if not bins:
    ax.hist(values, histtype="stepfilled", alpha=0.2)
  else:
    ax.hist(values, bins=bins, histtype="stepfilled", alpha=0.2)
  if logscale:
    plt.xscale("log")
  ax.set_ylabel("Occurrence Count")
  ax.set_xlabel(x_label)
  ax.set_title(title)
  plt.savefig(path)


def plot_heatmap(data: np.ndarray, path: str):
  """Plots and saves a heatmap of critical band co-occurrences."""
  fig, ax = plt.subplots()
  fig.set_size_inches(6.5, 6.5)
  ax.tick_params('both', labelsize=8)
  # We want to show all ticks...
  ax.set_xticks(np.arange(data.shape[1]))
  ax.set_yticks(np.arange(data.shape[0]))
  # ... and label them with the respective list entries.
  labels = ["CB {}".format(i) for i in range(data.shape[1])]
  ax.set_xticklabels(labels)
  ax.set_yticklabels(labels)
  ax.set_title("Count per combination of Critical Bands in Examples")
  sns.set(font_scale=0.6)
  sns.heatmap(data, annot=True, fmt="d", xticklabels=labels, yticklabels=labels,
              ax=ax)
  plt.savefig(path)


def plot_masking_patterns(curves: List[Dict[str, Any]],
                          save_directory: str):
  """Plots the data in Zwicker-style."""
  for masker_freq_probe_level in curves:
    masker_frequency = masker_freq_probe_level["masker_frequency"]
    probe_level = masker_freq_probe_level["probe_level"]
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_prop_cycle(color=[
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
        "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
        "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
        "#17becf", "#9edae5"
    ])
    plt.xscale("log")
    ax.set_xlabel("Probe Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Masked SPL (dB)", fontsize=12)
    ax.set_title("Masker Frequency {} Hz, Probe Level {} dB".format(
        masker_frequency, probe_level), fontsize=14)
    plt.axvline(x=masker_frequency, color="C2", ls=":")
    plt.axhline(y=probe_level, color="C2", ls=":")
    current_levels = []
    mean_masking_per_level = []
    for masker_curve in masker_freq_probe_level["curves"]:
      masker_level = masker_curve["masker_level"]
      current_levels.append(masker_level)
      frequencies = masker_curve["probe_frequencies"]
      masking = masker_curve["probe_masking"]
      if "failed" in masker_curve:
        new_frequencies = []
        new_masking = []
        for i, (freq, mask) in enumerate(zip(frequencies, masking)):
          if i not in masker_curve["failed"]:
            new_frequencies.append(freq)
            new_masking.append(mask)
        frequencies = new_frequencies
        masking = new_masking
      average_masking = [np.mean(np.array(m)) for m in masking]
      mean_masking_per_level.append(average_masking)
      std_masking = [np.std(np.array(m)) for m in masking]
      plt.errorbar(
          frequencies,
          average_masking,
          yerr=std_masking,
          fmt="o",
          label="Masker Level {}".format(masker_level))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    save_file = "masker_{}_probe_{}".format(masker_frequency, probe_level)
    with open(os.path.join(save_directory, save_file + ".txt"), "wt") as infile:
      for level, mean_masking in zip(current_levels, mean_masking_per_level):
        infile.write("level {}: ".format(level))
        freq_maskers_str = ";".join(["{},{}".format(f, m) for f, m in zip(
            frequencies, mean_masking)])
        infile.write(freq_maskers_str)
        infile.write("\n")
    plt.savefig(os.path.join(save_directory,
                             save_file + ".png"))


def plot_masking_patterns_threetone(curves: List[Dict[str, Any]],
                                    save_directory: str):
  """Plots the data in Zwicker-style."""
  for masker_freq_probe_level in curves:
    masker_frequencies = masker_freq_probe_level["masker_frequencies"]
    probe_level = masker_freq_probe_level["probe_level"]
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_prop_cycle(color=[
        "#ff7f0e", "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
        "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
        "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
        "#17becf", "#9edae5"
    ])
    plt.xscale("log")
    ax.set_xlabel("Probe Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Masked SPL (dB)", fontsize=12)
    ax.set_title("Masker Frequencies {} Hz, Probe Level {} dB".format(
        masker_frequencies, probe_level), fontsize=14)
    mask_freq_one, mask_freq_two = masker_frequencies.split("+")
    plt.axvline(x=float(mask_freq_one), color="C2", ls=":")
    plt.axvline(x=float(mask_freq_two), color="C2", ls=":")
    plt.axhline(y=probe_level, color="C2", ls=":")
    current_levels = []
    mean_masking_per_level = []
    for masker_curve in masker_freq_probe_level["curves"]:
      masker_level = masker_curve["masker_levels"]
      current_levels.append(masker_level)
      frequencies = masker_curve["probe_frequencies"]
      masking = masker_curve["probe_masking"]
      if "failed" in masker_curve:
        new_frequencies = []
        new_masking = []
        for i, (freq, mask) in enumerate(zip(frequencies, masking)):
          if i not in masker_curve["failed"]:
            new_frequencies.append(freq)
            new_masking.append(mask)
        frequencies = new_frequencies
        masking = new_masking
      average_masking = [np.mean(np.array(m)) for m in masking]
      mean_masking_per_level.append(average_masking)
      std_masking = [np.std(np.array(m)) for m in masking]
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




def plot_masking_patterns_grid(curves: List[Dict[str, Any]],
                               save_directory: str):
  """Plots the data in Zwicker-style."""
  masker_frequencies = list(set([c["masker_frequency"] for c in curves]))
  probe_levels = list(set([c["probe_level"] for c in curves]))
  masker_frequencies.sort()
  probe_levels.sort()
  freq_to_y_axis = {freq: i for i, freq in enumerate(masker_frequencies)}
  probe_to_x_axis = {probe: i for i, probe in enumerate(probe_levels)}
  fig, axs = plt.subplots(len(masker_frequencies),
                          len(probe_levels), figsize=(14, 20))
  plt.xscale("log")
  fig.tight_layout(pad=6.0)
  fig.text(0.5, 0.02, 'Probe Frequency (Hz)', ha='center', va='center', fontsize=14)
  fig.text(0.02, 0.5, 'Masked SPL (dB)', ha='center', va='center', rotation='vertical', fontsize=14)
  for i, masker_freq_probe_level in enumerate(curves):
    masker_frequency = masker_freq_probe_level["masker_frequency"]
    probe_level = masker_freq_probe_level["probe_level"]
    ax = axs[freq_to_y_axis[masker_frequency], probe_to_x_axis[probe_level]]
    ax.set_prop_cycle(color=[
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
        "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
        "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
        "#17becf", "#9edae5"
    ])
    ax.set_xscale("log")
    # ax.set_xlabel("Probe Frequency (Hz)", fontsize=8)
    # ax.set_ylabel("Masked SPL (dB)", fontsize=8)
    ax.set_title("Masker Frequency {} Hz, Probe Level {} dB".format(
        masker_frequency, probe_level), fontsize=12)
    ax.axvline(x=masker_frequency, color="C2", ls=":")
    ax.axhline(y=probe_level, color="C2", ls=":")
    current_levels = []
    mean_masking_per_level = []
    for masker_curve in masker_freq_probe_level["curves"]:
      masker_level = masker_curve["masker_level"]
      current_levels.append(masker_level)
      frequencies = masker_curve["probe_frequencies"]
      masking = masker_curve["probe_masking"]
      if "failed" in masker_curve:
        new_frequencies = []
        new_masking = []
        for j, (freq, mask) in enumerate(zip(frequencies, masking)):
          if j not in masker_curve["failed"]:
            new_frequencies.append(freq)
            new_masking.append(mask)
        frequencies = new_frequencies
        masking = new_masking
      average_masking = [np.mean(np.array(m)) for m in masking]
      mean_masking_per_level.append(average_masking)
      std_masking = [np.std(np.array(m)) for m in masking]
      ax.errorbar(
          frequencies,
          average_masking,
          yerr=std_masking,
          fmt="o",
          label="Masker Level {}".format(masker_level))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    save_file = "masker_{}_probe_{}".format(masker_frequency, probe_level)
    with open(os.path.join(save_directory, save_file + ".txt"), "wt") as infile:
      for level, mean_masking in zip(current_levels, mean_masking_per_level):
        infile.write("level {}: ".format(level))
        freq_maskers_str = ";".join(["{},{}".format(f, m) for f, m in zip(
            frequencies, mean_masking)])
        infile.write(freq_maskers_str)
        infile.write("\n")
  plt.savefig(os.path.join(save_directory,
                           "all_masking_patterns.png"))
