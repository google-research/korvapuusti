"""Test for loudness functions.
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
import unittest
import loudness


class LoudnessTest(unittest.TestCase):

  def test_loudness_roundtrip(self):
    for lo in (55, 62, 73, 84):
      for frequency in (58, 120, 223, 489, 4000, 9800):
        spl = loudness.loudness_to_spl(lo, frequency)
        lo_roundtrip = loudness.spl_to_loudness(spl, frequency)

        self.assertLess(abs(lo - lo_roundtrip), 5e-2)


if __name__ == '__main__':
  unittest.main()
