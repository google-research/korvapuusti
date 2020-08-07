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

import numpy as np
import math
import os
import scipy.stats
import scipy.integrate


class Model(object):

  def __init__(self):
    return

  def amplitude(self, learned_amplitude: float):
    return math.exp(learned_amplitude)

  def sigma(self, learned_sigma: float):
    return 250 * (1 + learned_sigma)

  def mean(self, learned_mean: float):
    return 499 + learned_mean

  def alpha(self, learned_alpha: float):
    return 4 + np.abs(learned_alpha)

  def loc(self, learned_loc: float):
    return 500 + learned_loc

  def scale(self, learned_scale: float):
    return np.abs(300 + learned_scale)

  @property
  def learned_parameters(self):
    return [float(os.environ["VAR0"]),
            float(os.environ["VAR1"]),
            float(os.environ["VAR2"]),
            float(os.environ["VAR3"])]

  def parameters_from_learned(self, parameters):
    amplitude = parameters[0]
    alpha = parameters[1]
    loc = parameters[2]
    scale = parameters[3]
    return [self.amplitude(amplitude), self.alpha(alpha), self.loc(loc),
            self.scale(scale)]

  def function(self, frequency, amp, alpha, loc, scale):
    integrate_left = -float("inf")
    integrate_right = alpha * ((frequency - loc) / scale)
    integrate_func = lambda t: (1 / np.sqrt(2 * np.pi)) * np.exp(
        -1 * (t**2 / 2))
    func = np.exp(-1 * ((frequency - loc)**2 / (2 * scale**2)))
    integrated = scipy.integrate.quad(integrate_func, integrate_left,
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
    return error

  def parameter_repr(self, parameters):
    parameters = tuple(self.parameters_from_learned(parameters))
    return "Amplitude: %.2f, Alpha: %.2f, Loc: %.2f, Scale: %.2f" % parameters
