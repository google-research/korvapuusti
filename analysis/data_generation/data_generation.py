# Lint as: python3
"""Generate data for listening tests.

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

import collections
import itertools
import random
from typing import List, Dict, Any

import data_analysis
import distributions
import loudness
import numpy as np
import scipy.stats

CbFrequencyPair = collections.namedtuple("CB_frequency_pair",
                                         ["cb", "frequency"])

ToneLevelPair = collections.namedtuple("Tone_level_pair", ["tone", "level"])
ProbeMaskerPair = collections.namedtuple("Probe_masker_pair",
                                         ["probe", "masker"])

# Estimated constants for beat range from listening tests.
FREQ_A, BEAT_RANGE_A = 100, 50
FREQ_B, BEAT_RANGE_B = 10000, 50
INTERPOLATE_SLOPE = (BEAT_RANGE_B - BEAT_RANGE_A) / (FREQ_B - FREQ_A)
ZERO_CROSSING = BEAT_RANGE_A - FREQ_A * INTERPOLATE_SLOPE

# Constants taken from ISO 226
ISO_FREQUENCIES = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
    12500, 20000
],
                           dtype=np.float64)


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


def calculate_beat_range(frequency: int):
  """Returns the beatrange based on a linear interpolation between constants."""
  return ZERO_CROSSING + INTERPOLATE_SLOPE * frequency


def check_frequencies(frequencies: np.ndarray, min_frequency: int,
                      max_frequency: int) -> bool:
  """Returns false when a pair of frequencies is too close together.

  When frequencies are to close is determined by the function
  calculate_beat_range and when a single frequency is too low
  or high.

  Args:
    frequencies: list of frequencies to be checked
    min_frequency: minimum frequency that is allowed
    max_frequency: maximum frequency that is allowed

  Returns:
    Boolean indicating whether a list of frequencies passes the constraints.
  """
  all_combinations = itertools.combinations(frequencies, 2)
  for freq_a, freq_b in all_combinations:
    # Get the minimum range freq_a and b need to be apart to not cause beating.
    beat_range_freq_a = calculate_beat_range(freq_a)
    beat_range_freq_b = calculate_beat_range(freq_b)
    beat_range = max(beat_range_freq_a, beat_range_freq_b)
    if abs(freq_a - freq_b) < beat_range:
      return False
  for freq in frequencies:
    if freq < min_frequency or freq > max_frequency:
      return False
  return True


def check_phons(phons: np.ndarray, min_phons: int, max_phons: int) -> bool:
  """Returns true if all phons levels are within [min_phons, max_phons]."""
  if np.all(phons >= min_phons) and np.all(phons <= max_phons):
    return True
  else:
    return False


def generate_iso_repro_examples(num_examples_per_curve: int,
                                num_curves: int,
                                clip_db=80) -> Dict[int, Dict[str, Any]]:
  """Generates a particular amount of examples per ISO equal loudness curve.

  Args:
    num_examples_per_curve: how many examples per curve to generate
    num_curves: how many equal loudness curves (phons) to generate examples from
    clip_db: maximum decibel level

  Returns:
    The data in a dict with as key the phons level (curve) and as values
    the reference tone (1000 Hz) level and a list of examples for that curve.

  """
  available_phons_levels = [20, 30, 40, 50, 60, 70, 80]
  assert num_curves < len(available_phons_levels), "Too many curves specified."
  phons_ptr = len(available_phons_levels) // 2
  chosen_phons_levels = []
  while len(chosen_phons_levels) < num_curves:
    chosen_phons_levels.append(available_phons_levels[phons_ptr])
    del available_phons_levels[phons_ptr]
    phons_ptr = len(available_phons_levels) // 2
  chosen_phons_levels.sort()

  # Find minimum frequency and phons level before it exceeds 90 dB
  max_phons = chosen_phons_levels[-1]
  frequency_ptr = -1
  spl = clip_db + 1
  while spl > clip_db and frequency_ptr < len(ISO_FREQUENCIES):
    frequency_ptr += 1
    spl = loudness.loudness_to_spl(max_phons, ISO_FREQUENCIES[frequency_ptr])
  min_frequency = ISO_FREQUENCIES[frequency_ptr]

  # Find maximum frequency and phons level before it exceeds 90 dB
  frequency_ptr = len(ISO_FREQUENCIES) // 2
  spl = 0
  while spl < clip_db and frequency_ptr < len(ISO_FREQUENCIES):
    spl = loudness.loudness_to_spl(max_phons, ISO_FREQUENCIES[frequency_ptr])
    frequency_ptr += 1
  max_frequency = ISO_FREQUENCIES[frequency_ptr - 1]
  if min_frequency > max_frequency:
    raise ValueError(
        "Can't generate examples below {} decibel for {} curves. "
        "Consider increasing clip_db or decreasing num_curves.".format(
            clip_db, num_curves))

  # Get the right number of frequencies with equal distance between them.
  frequency_steps = (np.log(max_frequency) -
                     np.log(min_frequency)) / num_examples_per_curve
  frequencies = [
      np.log(min_frequency) + frequency_steps * i
      for i in range(num_examples_per_curve)
  ]
  frequencies = [np.round(np.exp(frequency)) for frequency in frequencies]

  # Generate the data
  data = {}
  for phons_level in chosen_phons_levels:
    data[phons_level] = {
        "ref1000_spl": np.round(loudness.loudness_to_spl(phons_level, 1000)),
        "other_tones": []
    }
    for frequency in frequencies:
      spl = loudness.loudness_to_spl(phons_level, frequency)
      data[phons_level]["other_tones"].append({
          "frequency": frequency,
          "level": np.round(spl)
      })
  return data


def sample_n_per_cb(examples_per_cb: int,
                    critical_bands: List[int]) -> List[CbFrequencyPair]:
  """Samples examples_per_cb times a tone per critical band.

  Samples a specified number of integers for each critical band, where the
  critical bands are give in a list with adjacent pairs of integers giving
  the critical band boundaries, e.g., critical_bands[0] = 20,
  critical_bands[1] = 100, then the first cb is between [20, 99]
  Args:
    examples_per_cb: how many samples to get per cb
    critical_bands: the boundaries of each cb are at index i and i + 1

  Returns:
    a list of (cb, frequency) pairs
  """
  tones_per_cb = []
  for cb_i in range(len(critical_bands) - 1):
    for _ in range(examples_per_cb):
      sampled_tones = scipy.stats.randint.rvs(
          critical_bands[cb_i], critical_bands[cb_i + 1], size=1)
      tones_per_cb.append(CbFrequencyPair(cb_i, sampled_tones[0]))
  return tones_per_cb


def middle_frequency(left_cb: int, right_cb: int):
  return np.round((left_cb + right_cb) / 2)


def generate_two_tone_set(
    critical_bands: List[int], probe_levels: List[int],
    masker_levels: List[int], critical_bands_apart_probe: int,
    critical_bands_apart_masker: int, all_lower_probes: int,
    all_higher_probes: int) -> List[ProbeMaskerPair]:
  """Generates a small two-tone dataset.

  Generates a set of probe-masker pairs where each probe is a specified
  number of CBs apart and each masker as well. For each masker the
  parameters all_lower_probes and all_higher_probes specify which CBs around the
  masker CB will be taken as probes additionally.

  Args:
    critical_bands: the boundaries of each cb are at index i and i + 1
    probe_levels: which levels to generate probe tones with (SPL)
    masker_levels: which levels to generate masker tones with (SPL)
    critical_bands_apart_probe: how many CBs between each freq of probes
    critical_bands_apart_masker: how many CBs between each freq of maskers
    all_lower_probes: how many CBs below the masker CB should all be probes
    all_higher_probes: how many CBs above the masker CB should all be probes

  Returns:
    The data with tone combinations in a list.
  """
  if critical_bands_apart_probe >= len(critical_bands):
    raise ValueError("Can't be more CBs apart than there are CB's.")
  if critical_bands_apart_masker >= len(critical_bands):
    raise ValueError("Can't be more CBs apart than there are CB's.")

  examples = []

  # Precalculate the probe tones that will be used for every masker.
  probe_tones = []
  for cb_i in range(0, len(critical_bands) - 1, critical_bands_apart_probe):
    probe_tones.append(middle_frequency(critical_bands[cb_i],
                                        critical_bands[cb_i + 1]))
  # Generate level combinations that will be used for each probe-masker-pair.
  all_level_combinations = list(itertools.product(probe_levels, masker_levels))

  unique_examples = set()

  # Generate the examples.
  for cb_i_masker in range(0, len(critical_bands) - 1,
                           critical_bands_apart_masker):
    masker_f = middle_frequency(left_cb=critical_bands[cb_i_masker],
                                right_cb=critical_bands[cb_i_masker + 1])
    # Calculate the masker-dependent probe tones.
    left_cb_boundary = max(cb_i_masker - all_lower_probes, 0)
    right_cb_boundary = min(cb_i_masker + all_higher_probes + 1,
                            len(critical_bands) - 1)
    extra_probe_tones = [
        middle_frequency(critical_bands[cb], critical_bands[cb + 1])
        for cb in range(left_cb_boundary, right_cb_boundary)
    ]
    all_probes = set(probe_tones + extra_probe_tones)
    # Combine all probes with the current masker.
    for probe_f in all_probes:
      if probe_f == masker_f:
        continue
      for probe_level, masker_level in all_level_combinations:
        example_representation = "{},{},{},{}".format(probe_f, masker_f,
                                                      probe_level, masker_level)
        if example_representation in unique_examples:
          continue
        else:
          unique_examples.add(example_representation)
        examples.append(
            ProbeMaskerPair(
                masker=ToneLevelPair(tone=masker_f, level=masker_level),
                probe=ToneLevelPair(tone=probe_f, level=probe_level)))
  return examples


def generate_data(
    num_examples_per_cb: int,
    min_tones: int,
    max_tones: int,
    clip_db: int,
    desired_skewness: int,
    desired_mean: int,
    desired_variance: int,
    min_frequency: int,
    max_frequency: int,
    critical_bands: List[int],
    min_phons=0,
    max_phons=80,
    max_iterations=1000000) -> Dict[int, List[Dict[str, List[int]]]]:
  """Generates all listening data.

  Generates examples until each critical band has exactly
  `num_examples_per_cb` number of examples.
  TODO(lauraruis): check if n_per_cb cannot be generated (e.g.
  due to too large beatrange)

  Args:
    num_examples_per_cb: how many frequencies to sample per CB
    min_tones: minimum number of frequencies per example
    max_tones: maximum number of frequencies per example
    clip_db: maximum spl
    desired_skewness: skewness of skewed normal distr. for phons
    desired_mean: mean of skewed normal distr. for phons
    desired_variance: variance of skewed normal distr. for phons
    min_frequency: below this frequency no tones will be generated
    max_frequency: above this frequency no tones will be generated
    critical_bands: list of critical bands
    min_phons: minimum level of phons for examples
    max_phons: maximum level of phons for examples
    max_iterations: how long to try combining examples

  Returns:
    Dict with data examples.
  """
  # Initialize the structures for keeping the data.
  data = {i: [] for i in range(min_tones, max_tones + 1)}
  num_examples = 0

  # Calculate the needed shift and scaling for the SN distribution over phons.
  sn_shift, sn_scale = distributions.calculate_shift_scaling_skewnorm(
      desired_variance=desired_variance,
      desired_mean=desired_mean,
      alpha=desired_skewness)

  # Sample n examples per critical band.
  examples_per_cb = sample_n_per_cb(num_examples_per_cb, critical_bands)

  # Generate examples by combining a subset of tones until they run out.
  while len(examples_per_cb) >= min_tones:
    # Sample a number of tones for the example.
    num_tones = min(len(examples_per_cb), random.randint(min_tones, max_tones))

    # If less than max_tones examples left, check how far apart they are.
    if len(examples_per_cb) <= max_tones:
      if not check_frequencies(
          np.array([ex.frequency for ex in examples_per_cb]), min_frequency,
          max_frequency):
        break

    # Sample frequencies from the pre-generated tones until they satisfy the
    # constraint of being beat_range apart.
    frequencies = np.array([100] * num_tones)
    iteration = 0
    while not check_frequencies(frequencies, min_frequency,
                                max_frequency) and iteration < max_iterations:
      sampled_idxs = random.sample(range(len(examples_per_cb)), k=num_tones)
      for i, sampled_idx in enumerate(sampled_idxs):
        frequencies[i] = examples_per_cb[sampled_idx].frequency
      iteration += 1
    # If the correct frequencies weren't found, stop generation.
    if iteration >= max_iterations:
      print("WARNING: didn't find correct frequencies: ", frequencies)
      break

    # Delete the used tones from the pregenerated list.
    for sampled_idx in sorted(sampled_idxs, reverse=True):
      del examples_per_cb[sampled_idx]

    # Sample phons for each tone on a skewed normal distribution.
    phons = np.array([100] * num_tones)
    while not check_phons(phons, min_phons, max_phons):
      phons = distributions.sample_skewed_distribution(
          sn_shift, sn_scale, num_samples=num_tones, alpha=desired_skewness)
      phons = np.round(phons)
    num_examples += 1

    # Convert phons to sound pressure level in decibel with the ISO 226.
    sp_levels = [np.round(loudness.loudness_to_spl(
        loudness_phon=loudness_phons, frequency=frequency))
                 for loudness_phons, frequency in zip(phons, frequencies)]
    sp_levels = np.clip(np.array(sp_levels), a_min=0, a_max=clip_db)

    data[num_tones].append({
        "frequencies": list(frequencies),
        "phons": list(phons),
        "levels": list(sp_levels)
    })
  return data


def generate_extra_data(critical_bands: List[int],
                        sample_rate: int,
                        window_size: int,
                        cbs_below_masker: int,
                        cbs_above_masker: int,
                        minimum_db_masker: int,
                        minimum_db_probe: int,
                        maximum_db: int) -> List[ProbeMaskerPair]:
  """For each frequency bin of a FFT done with the specified window_size,
  sample a masker frequency from that bin and sample a probe frequency that
  is between cbs_below_masker and cbs_above_masker the masker CB."""
  examples = []
  num_bins = int(window_size / 2)
  for k in range(2, num_bins - 2):
    masker_frequency = data_analysis.bin_to_frequency(k, window_size,
                                                      sample_rate)

    if masker_frequency > 15000:
      return examples
    cb_masker = binary_search(critical_bands, masker_frequency)
    left_limit = max(0, cb_masker - cbs_below_masker)
    right_limit = min(len(critical_bands) - 2, cb_masker + cbs_above_masker)
    sampled_cbs_i = random.randint(left_limit, right_limit)
    sampled_cb_left = critical_bands[sampled_cbs_i]
    sampled_cb_right = critical_bands[sampled_cbs_i + 1]
    probe_frequency = random.uniform(sampled_cb_left, sampled_cb_right)
    masker_level = int(random.uniform(minimum_db_masker, maximum_db))
    probe_level = int(random.uniform(minimum_db_probe, masker_level))
    examples.append(
        ProbeMaskerPair(
            masker=ToneLevelPair(tone=masker_frequency, level=masker_level),
            probe=ToneLevelPair(tone=probe_frequency, level=probe_level)))
  return examples
