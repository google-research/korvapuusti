# Lint as: python3
"""Sampling from a SN and a uniform distribution.

Functionality to sample from a Skew Normal distribution and a continuous
log uniform distribution.

Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def calculate_shift_scaling_skewnorm(desired_variance: int, desired_mean: int,
                                     alpha: int) -> Tuple[float, float]:
  """Calculates the shift and scale to get a desired mean and variance.

  Calculate the desired scaling and shifting parameters to get a desired
  mean and variance from the Skew Normal Distribution.
  I.e., if X ∼ SN(0, 1, alpha), and Y = scale * X + shift, what scale and shift
  do we need to get the desired mean and variance of Y ∼ SK(shift, scale, alpha)
  Derived from: https://en.wikipedia.org/wiki/Skew_normal_distribution

  Args:
    desired_variance: the variance we want our variable to have
    desired_mean: the mean we want our variable to have
    alpha: the skewness parameter of the SN distribution

  Returns:
    The shift and scale parameters.
  """
  delta = (alpha / np.sqrt(1 + alpha**2))
  scaling = np.sqrt(desired_variance / (1 - (2 * delta**2) / np.pi))
  shift = desired_mean - scaling * delta * np.sqrt(2 / np.pi)
  return shift, scaling


def plot_skewed_distribution(shift: float,
                             scale: float,
                             path: str,
                             num_samples=1000,
                             alpha=-4):
  """Plots the distribution SN(shift, scaling**2, alpha)."""
  _, ax = plt.subplots(1, 1)
  x = np.linspace(
      scipy.stats.skewnorm.ppf(0.01, alpha, loc=shift, scale=scale),
      scipy.stats.skewnorm.ppf(0.99, alpha, loc=shift, scale=scale), 100)
  ax.plot(
      x,
      scipy.stats.skewnorm.pdf(x, alpha, loc=shift, scale=scale),
      "r-",
      lw=5,
      alpha=0.6,
      label="skewnorm pdf")
  mean, var = scipy.stats.skewnorm.stats(alpha, loc=shift, scale=scale)
  plt.title("SN - mean %.2f, var. %.2f, "
            "std.dev. %.2f" % (mean, var, np.sqrt(var)))
  r = scipy.stats.skewnorm.rvs(alpha, loc=shift, scale=scale, size=num_samples)
  ax.hist(r, density=True, histtype="stepfilled", alpha=0.2)
  ax.legend(loc="best", frameon=False)
  plt.savefig(path)
  return


def sample_skewed_distribution(shift: float, scale: float, alpha: int,
                               num_samples: int) -> np.ndarray:
  """Samples from num_samples from X ∼ SN(loc, scale**2, alpha)."""
  return scipy.stats.skewnorm.rvs(
      alpha, loc=shift, scale=scale, size=num_samples)


def calculate_shift_scaling_loguniform(desired_lowerbound: int,
                                       desired_upperbound: int,
                                       log_base: int) -> Tuple[float, float]:
  loc = math.log(desired_lowerbound, log_base)
  scale = math.log(desired_upperbound, log_base) - loc
  return loc, scale


def plot_uniform_distribution(num_samples: int, shift: float, scale: float,
                              path: str):
  """Plots the distribution U[shift, shift + scale]."""
  _, ax = plt.subplots(1, 1)
  x = np.linspace(scipy.stats.uniform.ppf(0.01, loc=shift, scale=scale),
                  scipy.stats.uniform.ppf(0.99, loc=shift, scale=scale), 100)
  ax.plot(x,
          scipy.stats.uniform.pdf(x, loc=shift, scale=scale),
          "r-", lw=5, alpha=0.6, label="uniform pdf")
  mean, var = scipy.stats.uniform.stats(moments="mv", loc=shift, scale=scale)
  plt.title("U - mean %.2f, var. %.2f, "
            "std.dev. %.2f" % (mean, var, np.sqrt(var)))
  r = scipy.stats.uniform.rvs(size=num_samples, loc=shift, scale=scale)
  ax.hist(r, density=True, histtype="stepfilled", alpha=0.2)
  ax.legend(loc="best", frameon=False)
  plt.savefig(path)
  return


def sample_log_distribution(num_samples: int, log_base: int, shift: float,
                            scale: float) -> np.ndarray:
  """Samples from a log-uniform distribution by log-scaling a uniform dist."""
  sampled_values = scipy.stats.uniform.rvs(
      size=num_samples, loc=shift, scale=scale)
  return log_base**sampled_values


def sample_uniform_distribution(num_samples: int, a: int, b: int):
  """Samples num_samples times uniformly between a and b."""
  loc = a
  scale = b - loc
  return scipy.stats.uniform.rvs(size=num_samples, loc=loc, scale=scale)
