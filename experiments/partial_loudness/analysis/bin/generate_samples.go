/* generate_sample_window generates a window of the audio used in evaluations and
 * stores the results in TFRecord files.
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
	"bufio"
	"encoding/json"
	"flag"
	"io"
	"log"
	"os"

	"github.com/google-research/korvapuusti/experiments/partial_loudness/analysis"
	"github.com/google-research/korvapuusti/tools/synthesize/signals"
	"github.com/ryszard/tfutils/go/tfrecord"
	"google.golang.org/protobuf/proto"

	proto1 "github.com/golang/protobuf/proto"
)

var (
	sampleRate                   = flag.Float64("sample_rate", 48000, "The sample rate to generate samples with.")
	windowSize                   = flag.Int("window_size", 2048, "The window size to generate.")
	playoutSeconds               = flag.Float64("playout_seconds", 1.0, "The number of seconds to play out before storing the last window.")
	evaluationJSON               = flag.String("evaluation_json", "", "Path to the file containing evaluations.")
	sampleOutput                 = flag.String("sample_output", "", "Path to the file the frequency JSON will be written to.")
	evaluationFullScaleSineLevel = flag.Float64("evaluation_full_scale_sine_level", 100, "dB SPL calibrated to a full scale sine in the evaluations.")
	sampleFullScaleSineLevel     = flag.Float64("sample_full_scale_sine_level", 100, "dB SPL calibrated to a full scale cale sine in the generated signal for FFT input.")
)

func main() {
	flag.Parse()
	if *evaluationJSON == "" || *sampleOutput == "" {
		flag.Usage()
		os.Exit(1)
	}

	evaluationFile, err := os.Open(*evaluationJSON)
	if err != nil {
		log.Panic(err)
	}
	defer evaluationFile.Close()
	sampleFile, err := os.Create(*sampleOutput)
	if err != nil {
		log.Panic(err)
	}
	defer sampleFile.Close()
	evaluationLines := bufio.NewReader(evaluationFile)
	for line, err := evaluationLines.ReadString('\n'); err == nil; line, err = evaluationLines.ReadString('\n') {
		evaluation := &analysis.EquivalentLoudness{}
		if err := json.Unmarshal([]byte(line), evaluation); err != nil {
			log.Panic(err)
		}
		if evaluation.EntryType == "EquivalentLoudnessMeasurement" {
			evaluation.Samples.FullScaleSineLevel = signals.DB(*sampleFullScaleSineLevel)
			evaluation.Samples.WindowSize = int64(*windowSize)
			evaluation.Samples.Rate = signals.Hz(*sampleRate)
			sampler, err := evaluation.Evaluation.Combined.Sampler()
			if err != nil {
				log.Panic(err)
			}
			signal, err := sampler.Sample(signals.TimeStretch{FromInclusive: 0, ToExclusive: signals.Seconds(*playoutSeconds)}, signals.Hz(*sampleRate), nil)
			if err != nil {
				log.Panic(err)
			}
			signal.AddLevel(signals.DB(*evaluationFullScaleSineLevel - *sampleFullScaleSineLevel))
			evaluation.Samples.Values = make([]float64, *windowSize)
			for idx := range evaluation.Samples.Values {
				evaluation.Samples.Values[idx] = signal[len(signal)-*windowSize+idx]
			}
			example, err := evaluation.ToTFExample()
			if err != nil {
				log.Panic(err)
			}
			encoded, err := proto.Marshal(proto1.MessageV2(example))
			if err != nil {
				log.Panic(err)
			}
			if err := tfrecord.Write(sampleFile, encoded); err != nil {
				log.Panic(err)
			}
		}
	}
	if err != nil && err != io.EOF {
		log.Panic(err)
	}
}
