# Lint as: python3
"""Tests for distributions.

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
import numpy as np
import scipy.stats

import distributions
import unittest


class DistributionsTest(unittest.TestCase):

  def test_calculate_shift_scaling_skewnorm_standardnormal(self):
    desired_variance = 0.40082844953639396
    desired_mean = 0.7740617226446519
    expected_scale = 1
    expected_shift = 0
    actual_shift, actual_scale = distributions.calculate_shift_scaling_skewnorm(
        desired_mean=desired_mean, desired_variance=desired_variance, alpha=4)
    self.assertAlmostEqual(expected_shift, actual_shift, places=7)
    self.assertAlmostEqual(expected_scale, actual_scale, places=7)

  def test_calculate_shift_scaling_skewnorm_upscale(self):
    desired_variance = 1.708168819476707
    desired_mean = 2.486120486787904
    expected_scale = 2
    expected_shift = 4
    actual_shift, actual_scale = distributions.calculate_shift_scaling_skewnorm(
        desired_mean=desired_mean, desired_variance=desired_variance, alpha=-3)
    self.assertAlmostEqual(expected_shift, actual_shift, places=7)
    self.assertAlmostEqual(expected_scale, actual_scale, places=7)

  def test_calculate_shift_scaling_skewnorm_neg_shift(self):
    desired_variance = 0.40082844953639396
    desired_mean = -4.225938277355348
    expected_scale = 1
    expected_shift = -5
    actual_shift, actual_scale = distributions.calculate_shift_scaling_skewnorm(
        desired_mean=desired_mean, desired_variance=desired_variance, alpha=4)
    self.assertAlmostEqual(expected_shift, actual_shift, places=7)
    self.assertAlmostEqual(expected_scale, actual_scale, places=7)

  def test_calculate_shift_scaling_skewnorm(self):
    desired_mean = 50
    desired_variance = 5
    shift, scale = distributions.calculate_shift_scaling_skewnorm(
        desired_mean=desired_mean, desired_variance=desired_variance, alpha=4)
    actual_mean, actual_variance = scipy.stats.skewnorm.stats(
        4, loc=shift, scale=scale)
    self.assertEqual(desired_mean, actual_mean)
    self.assertEqual(desired_variance, actual_variance)

  def test_sample_skewed_distribution(self):
    desired_mean = 50
    desired_variance = 5
    shift, scale = distributions.calculate_shift_scaling_skewnorm(
        desired_mean=desired_mean, desired_variance=desired_variance, alpha=4)
    sampled_values = distributions.sample_skewed_distribution(shift=shift,
                                                              scale=scale,
                                                              alpha=4,
                                                              num_samples=1000)
    actual_mean = np.mean(sampled_values)
    actual_variance = np.var(sampled_values)
    self.assertAlmostEqual(actual_mean, desired_mean, places=0)
    self.assertAlmostEqual(actual_variance, desired_variance, places=0)

  def test_sample_log_distribution(self):
    desired_lowerbound = 20
    desired_upperbound = 20000
    (shift, scale) = distributions.calculate_shift_scaling_loguniform(
        desired_upperbound=desired_upperbound,
        desired_lowerbound=desired_lowerbound,
        log_base=2)
    sampled_values = distributions.sample_log_distribution(10000, log_base=2,
                                                           shift=shift,
                                                           scale=scale)
    self.assertTrue(np.all(sampled_values <= desired_upperbound))
    self.assertTrue(np.all(sampled_values >= desired_lowerbound))

  def test_sample_uniform_distribution(self):
    desired_lowerbound = 20
    desired_upperbound = 20000
    sampled_values = distributions.sample_uniform_distribution(
        10000, a=desired_lowerbound, b=desired_upperbound)
    self.assertTrue(np.all(sampled_values <= desired_upperbound))
    self.assertTrue(np.all(sampled_values >= desired_lowerbound))


if __name__ == '__main__':
  unittest.main()
