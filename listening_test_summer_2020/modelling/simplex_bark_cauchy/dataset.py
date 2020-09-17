# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

"""TODO: DO NOT SUBMIT without one-line documentation for dataset.
"""
import os
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class MaskingCurve:
  masking_frequency: float
  masking_level: int
  probe_level: int
  probe_frequencies: List[float]
  decibel_masking: List[float]
  probe_frequencies_bark: List[float]
  bel_masking: List[float]


class MaskingDataset(object):

  def __init__(self):
    self._curves = {}
    self.critical_bands = [
        20, 100, 200, 300, 400, 505, 630, 770, 915, 1080, 1265, 1475, 1720,
        1990, 2310, 2690, 3125, 3675, 4350, 5250, 6350, 7650, 9400, 11750,
        15250, 20000
    ]

  def add_curve(self, curve: MaskingCurve):
    if curve.masking_frequency not in self._curves:
      self._curves[curve.masking_frequency] = {}
    if curve.probe_level not in self._curves[curve.masking_frequency]:
      self._curves[curve.masking_frequency][curve.probe_level] = {}
    self._curves[curve.masking_frequency][curve.probe_level][
        curve.masking_level] = curve

  @staticmethod
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

  def frequency_to_cb(self, frequency: float) -> int:
    return self.binary_search(self.critical_bands, frequency)

  def read_data(self, directory: str, input_file: str):
    metadata = input_file.strip(".txt").split("_")
    masking_frequency = metadata[1]
    probe_level = metadata[3]
    with open(os.path.join(directory, input_file), "r") as infile:
      for line in infile:
        split_line = line.split(":")
        masking_level = split_line[0].split()[1]
        data_points = split_line[1].split(";")
        probe_frequencies = []
        probe_frequencies_bark = []
        decibel_masking = []
        bel_masking = []
        for point in data_points:
          split_point = point.split(",")
          probe_frequency = float(split_point[0])
          probe_cb = self.frequency_to_cb(probe_frequency)
          probe_frequencies.append(probe_frequency)
          decibel_masking.append(float(split_point[1]))
          probe_frequencies_bark.append(probe_cb)
          bel_masking.append(float(split_point[1]) / 10)
        masking_curve = MaskingCurve(float(masking_frequency),
                                     int(masking_level),
                                     int(probe_level),
                                     probe_frequencies,
                                     decibel_masking,
                                     probe_frequencies_bark,
                                     bel_masking)
        self.add_curve(masking_curve)
    return

  def get_curve_data(self, masking_frequency: float, probe_level: int,
                     masking_level: int):
    if masking_frequency not in self._curves:
      raise ValueError(
          "No curve for masking frequency {}".format(masking_frequency))
    if probe_level not in self._curves[masking_frequency]:
      raise ValueError("No curve for probe level {}".format(probe_level))
    if masking_level not in self._curves[masking_frequency][probe_level]:
      raise ValueError("No curve for masking level {}".format(masking_level))
    curve = self._curves[masking_frequency][probe_level][masking_level]
    data = list(zip(curve.probe_frequencies_bark, curve.bel_masking))
    return data

  def get_all_data(self):
    for masking_frequency, probe_masker_curves in self._curves.items():
      for probe_level, masker_curves in probe_masker_curves.items():
        for masker_level, curve_data in masker_curves.items():
          for (probe_frequency_bark, bel_masking, probe_frequency) in zip(curve_data.probe_frequencies_bark,
                                                        curve_data.bel_masking,
                                                         curve_data.probe_frequencies):
            yield {"masker_frequency": masking_frequency,
                   "probe_level": probe_level,
                   "probe_frequency": probe_frequency,
                   "masker_level": masker_level,
                   "probe_frequency_bark": probe_frequency_bark,
                   "target_bel_masking": bel_masking}
