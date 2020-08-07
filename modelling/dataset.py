"""TODO: DO NOT SUBMIT without one-line documentation for dataset.
"""
import os
from typing import List
from dataclasses import dataclass


@dataclass
class MaskingCurve:
  masking_frequency: float
  masking_level: int
  probe_level: int
  probe_frequencies: List[float]
  decibel_masking: List[float]


class MaskingDataset(object):

  def __init__(self):
    self._curves = {}

  def add_curve(self, curve: MaskingCurve):
    if curve.masking_frequency not in self._curves:
      self._curves[curve.masking_frequency] = {}
    if curve.probe_level not in self._curves[curve.masking_frequency]:
      self._curves[curve.masking_frequency][curve.probe_level] = {}
    self._curves[curve.masking_frequency][curve.probe_level][
        curve.masking_level] = curve

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
        decibel_masking = []
        for point in data_points:
          split_point = point.split(",")
          probe_frequencies.append(float(split_point[0]))
          decibel_masking.append(float(split_point[1]))
        masking_curve = MaskingCurve(float(masking_frequency),
                                     int(masking_level),
                                     int(probe_level),
                                     probe_frequencies,
                                     decibel_masking)
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
    data = list(
        zip(
            self._curves[masking_frequency][probe_level]
            [masking_level].probe_frequencies, self._curves[masking_frequency]
            [probe_level][masking_level].decibel_masking))
    return data
