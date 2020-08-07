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

"""TODO(lauraruis): DO NOT SUBMIT without one-line documentation for loss.

TODO(lauraruis): DO NOT SUBMIT without a detailed description of loss.
"""

import os

import dataset
import model


dataset = dataset.MaskingDataset()
for filename in os.listdir("data"):
  if filename.startswith("masker") and filename.endswith(".txt"):
    dataset.read_data("data", filename)

masking_frequency = float(os.environ["MASK_FREQ"])
probe_level = int(os.environ["PROBE_LEVEL"])
masking_level = int(os.environ["MASKING_LEVEL"])
# masking_frequency = 568.0
# probe_level = 60
# masking_level = 80
data = dataset.get_curve_data(masking_frequency=masking_frequency,
                              probe_level=probe_level,
                              masking_level=masking_level)
model_class = model.Model()

print("Loss", model_class.aggregate_loss(data))
print("Current Best: ", os.environ["BEST"])
