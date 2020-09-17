/* data contains definitions of data formats for the partial_loudness
 * experiment.
 *
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 */
package analysis

import (
	"github.com/google-research/korvapuusti/tools/spectrum"
	"github.com/google-research/korvapuusti/tools/synthesize/signals"
)

type EquivalentLoudness struct {
	EntryType   string
	Calibration struct {
		HeadphoneFrequencyResponseHash string
		FullScaleSineDBSPL             float64
	}
	Run struct {
		ID string
	}
	Evaluation struct {
		ID        string
		Frequency float64
		Probe     signals.SamplerWrapper
		Combined  signals.SamplerWrapper
	}
	Results struct {
		ProbeGainForEquivalentLoudness  float64
		ProbeDBSPLForEquivalentLoudness float64
	}
	Analysis struct {
		Channels         [][]float64
		ChannelSpectrums []spectrum.S
	}
}
