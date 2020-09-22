/* generate_snr runs CARFAC on the audio used in evaluations, calculate the SNR
 * of the CARFAC output, and stores the results in JSON files.
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
	"github.com/google-research/korvapuusti/tools/carfac"
	"github.com/google-research/korvapuusti/tools/spectrum"
	"github.com/google-research/korvapuusti/tools/synthesize/signals"
	"github.com/ryszard/tfutils/go/tfrecord"
	"google.golang.org/protobuf/proto"

	proto1 "github.com/golang/protobuf/proto"
)

const (
	rate          = 48000
	fftWindowSize = 2048
)

var (
	evaluationJSON = flag.String("evaluation_json", "", "Path to the file containing evaluations.")
	snrOutput      = flag.String("snr_output", "", "Path to the file the SNR JSON will be written to.")
)

func main() {
	flag.Parse()
	if *evaluationJSON == "" || *snrOutput == "" {
		flag.Usage()
		os.Exit(1)
	}

	cf := carfac.New(rate)

	evaluationFile, err := os.Open(*evaluationJSON)
	if err != nil {
		log.Panic(err)
	}
	defer evaluationFile.Close()
	snrFile, err := os.Create(*snrOutput)
	if err != nil {
		log.Panic(err)
	}
	defer snrFile.Close()
	evaluationLines := bufio.NewReader(evaluationFile)
	for line, err := evaluationLines.ReadString('\n'); err == nil; line, err = evaluationLines.ReadString('\n') {
		evaluation := &analysis.EquivalentLoudness{}
		if err := json.Unmarshal([]byte(line), evaluation); err != nil {
			log.Panic(err)
		}
		if evaluation.EntryType == "EquivalentLoudnessMeasurement" {
			sampler, err := evaluation.Evaluation.Combined.Sampler()
			if err != nil {
				log.Panic(err)
			}
			signal, err := sampler.Sample(signals.TimeStretch{FromInclusive: 0, ToExclusive: 1}, rate, nil)
			if err != nil {
				log.Panic(err)
			}
			carfacInput := make([]float32, cf.NumSamples())
			for idx := range carfacInput {
				carfacInput[idx] = float32(signal[len(signal)-len(carfacInput)+idx])
			}
			cf.Run(carfacInput)
			nap, err := cf.NAP()
			if err != nil {
				log.Panic(err)
			}
			for chanIdx := 0; chanIdx < cf.NumChannels(); chanIdx++ {
				channel := make([]float64, fftWindowSize)
				for sampleIdx := range channel {
					channel[sampleIdx] = float64(nap[(cf.NumSamples()-fftWindowSize+sampleIdx)*cf.NumChannels()+chanIdx])
				}
				evaluation.Analysis.Channels = append(evaluation.Analysis.Channels, channel)
				evaluation.Analysis.ChannelSpectrums = append(evaluation.Analysis.ChannelSpectrums, spectrum.Compute(channel, rate))
			}
			example, err := evaluation.ToTFExample()
			if err != nil {
				log.Panic(err)
			}
			encoded, err := proto.Marshal(proto1.MessageV2(example))
			if err != nil {
				log.Panic(err)
			}
			if err := tfrecord.Write(snrFile, encoded); err != nil {
				log.Panic(err)
			}
		}
	}
	if err != nil && err != io.EOF {
		log.Panic(err)
	}
}
