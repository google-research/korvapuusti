"""ISO 226 Equal-loudness contours.
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
from scipy import interpolate


# Constants taken from ISO 226
_f = np.array(
    [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
     630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
     10000, 12500, 20000], dtype=np.float64)

_af = np.array(
    [0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330,
     0.315, 0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244,
     0.243, 0.243, 0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301, 0.271],
    dtype=np.float64)

_lu = np.array(
    [-31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5,
     -3.1, -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7,
     2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1, -10.7],
    dtype=np.float64)

_tf = np.array(
    [78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4,
     11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0,
     -5.4, -1.5, 6.0, 12.6, 13.9, 12.3, 13.9],
    dtype=np.float64)


_coeffs = np.stack([_af, _lu, _tf], axis=1)
_coeffs = interpolate.CubicSpline(_f, _coeffs, axis=0)


def spl_to_loudness(spl_db, frequency):
  """Converts sound pressure to loudness at a given frequency.

  ISO 226 Equal-Loudness conversion of sound pressure to loudness.
  Computes the loudness of a pure sine wave of the given sound pressure
  at the given frequency.

  Args:
    spl_db: the sound pressure in dB.
    frequency: the frequency of the signal.

  Returns:
    the perceived sound loudness in phons.
  """
  af, lu, tf = np.moveaxis(_coeffs(frequency), -1, 0)

  def expf(x):
    return (0.4 * 10 ** ((x + lu) / 10 - 9)) ** af

  bf = expf(spl_db) - expf(tf) + 0.005135
  return 40 * np.log10(bf) + 94


def loudness_to_spl(loudness_phon, frequency):
  """Converts loudness to sound pressure at a given frequency.

  ISO 226 Equal-Loudness conversion of loudness to sound pressure.
  Computes the sound pressure of a pure sine wave of the given loudness
  at the given frequency.

  Args:
    loudness_phon: the loudness in phons.
    frequency: the frequency of the signal.

  Returns:
    the sound pressure in dB.
  """
  alpha_f, lu, tf = np.moveaxis(_coeffs(frequency), -1, 0)

  af = 4.47e-3 * (10 ** (0.025 * loudness_phon) - 1.15)
  af = af + (0.4 * 10 ** ((tf + lu) / 10 - 9)) ** alpha_f

  return (10 / alpha_f) * np.log10(af) - lu + 94


def rescale_loudness_amplitude(loudness_phon, frequency, reference_spl=90):
  """Computes the scaling required to obtain the desired loudness.

  Args:
    loudness_phon: the desired loudness in phons.
    frequency: the frequency of the signal.
    reference_spl: the reference sound pressure when the scaling is 1.

  Returns:
    A scalar representing the desired scaling.
  """
  spec_spl = loudness_to_spl(loudness_phon, frequency)
  spl_diff = spec_spl - reference_spl

  if np.any(spl_diff > 0):
    print('rescaling caused pressure to go above reference')
    spl_diff = np.minimum(spl_diff, 0)

  return 10 ** (spl_diff / 20)

