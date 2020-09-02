/* The synthesize command synthesizes audio signals according to provided specs.
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
package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"
	"os"

	"github.com/google-research/korvapuusti/tools/synthesize/signals"
)

var (
	signalSpec                   = flag.String("signal_spec", "", "The signal to synthesize, given as a SamplerWrapper JSON.")
	speakerFrequencyResponseFile = flag.String("speaker_frequency_response_file", "", "The file containing the frequency response of the speakers, produced by the calibrate/calibrate.html tool.")
	sampleRate                   = flag.Float64("sample_rate", 48000.0, "Sample rate to use when synthesizing.")
	durationSeconds              = flag.Float64("duration_seconds", 1.0, "Number of seconds to synthesize.")
	destination                  = flag.String("destination", "", "WAV file to store the synthesized buffer in.")
)

func main() {
	flag.Parse()
	if *signalSpec == "" || *destination == "" {
		flag.Usage()
		os.Exit(1)
	}

	signal, err := signals.ParseSampler(*signalSpec)
	if err != nil {
		panic(err)
	}

	var speakerFrequencyResponse signals.FrequencyResponse
	if *speakerFrequencyResponseFile != "" {
		blob, err := ioutil.ReadFile(*speakerFrequencyResponseFile)
		if err != nil {
			panic(err)
		}
		measurements := []map[string]float64{}
		if err := json.Unmarshal(blob, &measurements); err != nil {
			panic(err)
		}
		speakerFrequencyResponse, err = signals.LoadCalibrateFrequencyResponse(measurements)
		if err != nil {
			panic(err)
		}
	}

	writer, err := os.Create(*destination)
	if err != nil {
		panic(err)
	}
	defer writer.Close()

	samples, err := signal.Sample(signals.TimeStretch{0, signals.Seconds(*durationSeconds)}, signals.Hz(*sampleRate), speakerFrequencyResponse)
	if err != nil {
		panic(err)
	}

	if err := samples.WriteWAV(writer, *sampleRate); err != nil {
		panic(err)
	}
}
