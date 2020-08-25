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

import os
from typing import List

import numpy as np
import scipy.integrate
import scipy.stats


class FullModel(object):

  def __init__(self):
    self.models = {}
    self.mask_frequencies = []
    self.probe_levels = []
    self.mask_levels = []
    return

  def initialize_models(self, vars_path: str):

    with open(vars_path, "r") as infile:
      for i, line in enumerate(infile):
        if i == 0:
          continue
        splitted_line = line.split(",")
        self.mask_frequencies.append(float(splitted_line[0]))
        self.probe_levels.append(float(splitted_line[1]))
        self.mask_levels.append(float(splitted_line[2]))
        model_class = Model(
            float(splitted_line[0]), int(splitted_line[1]),
            int(splitted_line[2]))
        actual_parameters = model_class.parameters_from_learned(
            [float(entry) for entry in splitted_line[4:]])
        model_class.learned_pars = actual_parameters
        self.add_model(model_class)

  def add_model(self, model):
    if model.masker_frequency not in self.models:
      self.models[model.masker_frequency] = {}
    if model.masker_level not in self.models[model.masker_frequency]:
      self.models[model.masker_frequency][model.masker_level] = {}
    self.models[model.masker_frequency][model.masker_level][model.probe_level] = model
    return

  def find_closest(self, value, array_to_search):
    array_to_search.sort()
    if value <= array_to_search[0]:
      return array_to_search[0]
    elif value >= array_to_search[-1]:
      return array_to_search[-1]
    else:
      for i in range(0, len(array_to_search) - 1):
        if value >= array_to_search[i] and value <= array_to_search[i + 1]:
          diff_left = value - array_to_search[i]
          diff_right = array_to_search[i + 1] - value
          if diff_left < diff_right:
            return array_to_search[i]
          else:
            return array_to_search[i + 1]
    return -1

  def find_closest_model(self, masker_frequency: float, probe_level: int,
                         masker_level: int):
    current_masker_frequencies = list(self.models.keys())
    current_masker_frequencies.sort()
    if isinstance(masker_frequency, float):
      closest_frequency = self.find_closest(masker_frequency,
                                            current_masker_frequencies)
      current_masker_levels = list(self.models[closest_frequency])
      closest_masker_level = self.find_closest(masker_level,
                                               current_masker_levels)
      current_probe_levels = list(self.models[closest_frequency][closest_masker_level])
      closest_probe_level = self.find_closest(probe_level,
                                              current_probe_levels)
      selected_model = self.models[closest_frequency][closest_masker_level][closest_probe_level]
    else:
      masker_one, masker_two = masker_frequency.split("+")
      masker_one = float(masker_one)
      masker_two = float(masker_two)
      selected_model_one = self.find_closest_model(masker_one, probe_level,
                                                   masker_level)
      selected_model_two = self.find_closest_model(masker_two, probe_level,
                                                   masker_level)
      selected_model = (selected_model_one, selected_model_two)
    return selected_model

  def predict(self, masker_frequency: float, probe_level: int,
              masker_level: int, probe_frequency: float):
    selected_model = self.find_closest_model(masker_frequency,
                                             probe_level,
                                             masker_level)
    if not isinstance(selected_model, tuple):
      selected_model.update_model(masker_frequency, probe_level,
                                  masker_level)
      model_inputs = [probe_frequency] + selected_model.learned_pars
      prediction = selected_model.function(*model_inputs)
    else:
      masker_one, masker_two = masker_frequency.split("+")
      masker_one = float(masker_one)
      masker_two = float(masker_two)
      model_one = selected_model[0]
      model_one.update_model(masker_one, probe_level, masker_level)
      model_one_inputs = [probe_frequency] + model_one.learned_pars
      prediction_one = model_one.function(*model_one_inputs)
      model_two = selected_model[1]
      model_two.update_model(masker_two, probe_level, masker_level)
      model_two_inputs = [probe_frequency] + model_two.learned_pars
      prediction_two = model_two.function(*model_two_inputs)
      prediction = min(prediction_one + prediction_two, probe_level)

    return prediction


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

def frequency_to_cb(frequency: float, critical_bands: List[int]) -> int:
  return binary_search(critical_bands, frequency)


class Model(object):

  def __init__(self, masker_frequency: float, probe_level: int,
               masker_level: int):
    self.masker_frequency = masker_frequency
    self.probe_level = probe_level
    self.masker_level = masker_level
    self.critical_bands = [
        20, 100, 200, 300, 400, 505, 630, 770, 915, 1080, 1265, 1475, 1720,
        1990, 2310, 2690, 3125, 3675, 4350, 5250, 6350, 7650, 9400, 11750,
        15250, 20000
    ]
    self.masker_frequency_bark = frequency_to_cb(masker_frequency,
                                                 self.critical_bands)
    self.learned_pars = None
    self.constant_alpha = 3
    self.allowed_amp = max(min(
        self.probe_level / 10,
        np.abs((self.masker_level / 10) - (self.probe_level / 10))), 1)
    return

  def update_model(self, new_masker_frequency: float, new_probe_level: int,
                   new_masker_level: int):
    self.masker_frequency_bark = frequency_to_cb(new_masker_frequency,
                                                 self.critical_bands)
    self.masker_frequency = new_masker_frequency
    self.probe_level = new_probe_level
    self.masker_level = new_masker_level
    self.allowed_amp = max(min(
        self.probe_level / 10,
        np.abs((self.masker_level / 10) - (self.probe_level / 10))), 1)

  def amplitude(self, learned_amplitude: float):
    return self.allowed_amp - learned_amplitude

  def delta(self, alpha: float) -> float:
    numerator = alpha
    denominator = np.sqrt(1 + alpha**2)
    return numerator / denominator

  def mean_z(self, alpha: float) -> float:
    return np.sqrt(2 / np.pi) * self.delta(alpha)

  def sigma_z(self, alpha: float) -> float:
    return np.sqrt(1 - self.mean_z(alpha)**2)

  def skewness(self, alpha: float) -> float:
    delta = self.delta(alpha)
    constant = (4 - np.pi) / 2
    numerator = (delta * np.sqrt(2 / np.pi))**3
    denominator = (1 - (2 * delta**2) / np.pi)**(3 / 2)
    return constant * (numerator / denominator)

  def numerical_mode(self, alpha: float) -> float:
    term_one = (self.skewness(alpha) * self.sigma_z(alpha)) / 2
    term_two = (np.sign(alpha) / 2) * np.exp(
        -1 * (2 * np.pi) / (np.abs(alpha) + 1e-5))
    return self.mean_z(alpha) - term_one - term_two

  def location(self, alpha: float, scale: float):
    return self.masker_frequency_bark - scale * self.numerical_mode(alpha)

  def alpha(self, learned_alpha: float):
    return self.constant_alpha + learned_alpha

  def scale(self, learned_scale: float):
    return 1 + learned_scale

  @property
  def learned_parameters(self):
    return [float(os.environ["VAR0"]),
            float(os.environ["VAR1"]),
            float(os.environ["VAR2"])]

  def parameters_from_learned(self, parameters):
    amplitude = parameters[0]
    alpha = parameters[1]
    scale = parameters[2]
    return [self.amplitude(amplitude), self.alpha(alpha), self.scale(scale)]

  def to_integrate(self, t):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-1 * (t**2 / 2))

  def function(self, frequency, amp, alpha, scale):
    loc = self.location(alpha, scale)
    # Don't use -inf because of numerical instabilities.
    # https://stackoverflow.com/questions/24003694/discontinuity-in-results-when-using-scipy-integrate-quad
    integrate_left = -5
    integrate_right = alpha * ((frequency - loc) / (scale + 1e-6))
    func = np.exp(-1 * ((frequency - loc)**2 / (2 * scale**2 + 1e-6)))
    integrated = scipy.integrate.quad(self.to_integrate, integrate_left,
                                      integrate_right)
    sample = func * integrated[0]
    return amp * sample

  def aggregate_loss(self, points):
    error = 0
    for frequency, actual_amp in points:
      inputs = [frequency] + self.parameters_from_learned(
          self.learned_parameters)
      predicted_amplitude = self.function(*inputs)
      current_error = (actual_amp - predicted_amplitude)**2
      error += current_error
    # Penalty for too high amplitude
    learned_amplitude = self.learned_parameters[0]
    learned_var_scale = self.scale(self.learned_parameters[2])
    if learned_amplitude < 0:
      error += np.abs(learned_amplitude)
    if learned_amplitude > self.allowed_amp:
      error += np.abs(learned_amplitude - self.allowed_amp)
    return error

  def parameter_repr(self, parameters):
    parameters = tuple(self.parameters_from_learned(parameters))
    return "Amplitude: %.2f, Alpha: %.2f, Scale: %.2f" % parameters
