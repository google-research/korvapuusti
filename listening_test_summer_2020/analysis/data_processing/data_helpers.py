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

from typing import Any, Dict


class AnswerLookupTable():
  """Lookup Table for finding perceived probe levels from CC answers."""

  def __init__(self):
    self.table = {}

  def get_table_key(self, masker_frequency: float,
                    probe_level: int,
                    masker_level: int):
    return "{},{},{}".format(str(masker_frequency),
                             str(probe_level),
                             str(masker_level))

  def add(self, masker_frequency: float, probe_level: int, masker_level: int,
          probe_frequency: float, example: Dict[str, Any]):
    table_key = self.get_table_key(masker_frequency, probe_level, masker_level)
    if table_key not in self.table.keys():
      self.table[table_key] = {}
    self.table[table_key][probe_frequency] = example

  def extract(self, masker_frequency: float, probe_level: int,
              masker_level: int, probe_frequency: float):
    table_key = self.get_table_key(masker_frequency, probe_level, masker_level)
    if self.table.get(table_key):
      return self.table[table_key].get(probe_frequency)
    else:
      return None
