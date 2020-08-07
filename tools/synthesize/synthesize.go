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
	"flag"
	"os"
	"strings"

	"github.com/google-research/korvapuusti/tools/synthesize/signals"
)

var (
	signalSpecs = flag.String("signal_specs", "", `The signals to synthesize, given as separate arguments.
Has to follow the form of 'SPEC,...,SPEC' where each SPEC has to follow the format in signals/signals.go.`)
	sampleRate      = flag.Float64("sample_rate", 48000.0, "Sample rate to use when synthesizing.")
	durationSeconds = flag.Float64("duration_seconds", 1.0, "Number of seconds to synthesize.")
	destination     = flag.String("destination", "", "WAV file to store the synthesized buffer in.")
)

func main() {
	flag.Parse()
	if *signalSpecs == "" || *destination == "" {
		flag.Usage()
		os.Exit(1)
	}

	super := signals.Superposition{}
	for _, spec := range strings.Split(*signalSpecs, ",") {
		signal, err := signals.ParseSampler(spec)
		if err != nil {
			panic(err)
		}
		super = append(super, signal)
	}

	writer, err := os.Create(*destination)
	if err != nil {
		panic(err)
	}
	defer writer.Close()

	samples, err := super.Sample(signals.TimeStretch{0, signals.Seconds(*durationSeconds)}, signals.Hz(*sampleRate))
	if err != nil {
		panic(err)
	}

	if err := samples.WriteWAV(writer, *sampleRate); err != nil {
		panic(err)
	}
}
