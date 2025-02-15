/* generate_snr runs CARFAC on the audio used in evaluations, calculate the SNR
 * of the CARFAC output, and stores the results in TFRecord files.
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

	"github.com/cheggaaa/pb"
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
	evaluationJSON               = flag.String("evaluation_json", "", "Path to the file containing evaluations.")
	snrOutput                    = flag.String("snr_output", "", "Path to the file the SNR JSON will be written to.")
	evaluationFullScaleSineLevel = flag.Float64("evaluation_full_scale_sine_level", 100, "dB SPL calibrated to a full scale sine in the evaluations.")
	carfacFullScaleSineLevel     = flag.Float64("carfac_full_scale_sine_level", 100, "dB SPL for a full scale sine in the generated signal for CARFAC input.")
	carfacZeroVOffset            = flag.Bool("carfac_zero_v_offset", false, "Whether to zero the v_offset CARFAC parameter as mentioned in https://asa.scitation.org/doi/10.1121/1.5038595.")
	carfacOpenLoop               = flag.Bool("carfac_open_loop", false, "Whether to run CARFAC on the generated samples an extra time with an open loop before analysing the results.")
	carfacERBPerStep             = flag.Float64("carfac_erb_per_step", 0.0, "Custom erb_per_step when running CARFAC. 0.0 means use default value.")
	carfacMaxZeta                = flag.Float64("carfac_max_zeta", 0.0, "Custom max_zeta when running CARFAC. 0.0 means use default value.")
	carfacZeroRatio              = flag.Float64("carfac_zero_ratio", 0.0, "Custom zero_ratio when running CARFAC. 0.0 means use default value.")
	carfacStageGain              = flag.Float64("carfac_stage_gain", 0.0, "Custom agc_stage_gain when running CARFAC. 0.0 means use default value.")
	noiseFloor                   = flag.Float64("noise_floor", 35, "dB SPL of noise where evaluations were made.")
)

func main() {
	flag.Parse()
	if *evaluationJSON == "" || *snrOutput == "" {
		flag.Usage()
		os.Exit(1)
	}

	var vOffset *float64
	if *carfacZeroVOffset {
		zero := 0.0
		vOffset = &zero
	}
	var erbPerStep *float64
	if *carfacERBPerStep != 0.0 {
		erbPerStep = carfacERBPerStep
	}
	var maxZeta *float64
	if *carfacMaxZeta != 0.0 {
		maxZeta = carfacMaxZeta
	}
	var zeroRatio *float64
	if *carfacZeroRatio != 0.0 {
		zeroRatio = carfacZeroRatio
	}
	var stageGain *float64
	if *carfacStageGain != 0.0 {
		stageGain = carfacStageGain
	}
	cf := carfac.New(carfac.CARFACParams{SampleRate: rate, VOffset: vOffset, ERBPerStep: erbPerStep, MaxZeta: maxZeta, ZeroRatio: zeroRatio, StageGain: stageGain})

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

	lineReader := bufio.NewReader(evaluationFile)
	evaluations := []*analysis.EquivalentLoudness{}
	for line, err := lineReader.ReadString('\n'); err == nil; line, err = lineReader.ReadString('\n') {
		evaluation := &analysis.EquivalentLoudness{}
		if err := json.Unmarshal([]byte(line), evaluation); err != nil {
			log.Panic(err)
		}
		if evaluation.EntryType == "EquivalentLoudnessMeasurement" {
			evaluations = append(evaluations, evaluation)
		}
	}
	if err != nil && err != io.EOF {
		log.Panic(err)
	}

	bar := pb.StartNew(len(evaluations))
	for _, evaluation := range evaluations {
		evaluation.Analysis.CARFACFullScaleSineLevel = signals.DB(*carfacFullScaleSineLevel)
		evaluation.Analysis.OpenLoop = *carfacOpenLoop
		evaluation.Analysis.VOffsetProvided = *carfacZeroVOffset
		evaluation.Analysis.ERBPerStep = *carfacERBPerStep
		evaluation.Analysis.NoiseFloor = signals.DB(*noiseFloor)
		evaluation.Analysis.MaxZeta = *carfacMaxZeta
		evaluation.Analysis.ZeroRatio = *carfacZeroRatio
		evaluation.Analysis.StageGain = *carfacStageGain
		if *carfacZeroVOffset {
			evaluation.Analysis.VOffset = *vOffset
		}
		sampler, err := evaluation.Evaluation.Combined.Sampler()
		if err != nil {
			log.Panic(err)
		}
		noisySampler := signals.Superposition{sampler, &signals.Noise{Color: signals.White, LowerLimit: 20, UpperLimit: 20000, Level: signals.DB(*noiseFloor - *evaluationFullScaleSineLevel)}}
		signal, err := noisySampler.Sample(signals.TimeStretch{FromInclusive: 0, ToExclusive: 1}, rate, nil)
		if err != nil {
			log.Panic(err)
		}
		signal.AddLevel(signals.DB(*evaluationFullScaleSineLevel - *carfacFullScaleSineLevel))
		carfacInput := make([]float32, cf.NumSamples())
		for idx := range carfacInput {
			carfacInput[idx] = float32(signal[len(signal)-len(carfacInput)+idx])
		}
		cf.Run(carfacInput)
		if *carfacOpenLoop {
			cf.RunOpen(carfacInput)
		}

		nap, err := cf.NAP()
		if err != nil {
			log.Panic(err)
		}
		for chanIdx := 0; chanIdx < cf.NumChannels(); chanIdx++ {
			channel := make([]float64, fftWindowSize)
			for sampleIdx := range channel {
				channel[sampleIdx] = float64(nap[(cf.NumSamples()-fftWindowSize+sampleIdx)*cf.NumChannels()+chanIdx])
			}
			evaluation.Analysis.NAPChannels = append(evaluation.Analysis.NAPChannels, channel)
			evaluation.Analysis.NAPChannelSpectrums = append(evaluation.Analysis.NAPChannelSpectrums, spectrum.Compute(channel, rate))

			evaluation.Analysis.ChannelPoles = append(evaluation.Analysis.ChannelPoles, float64(cf.Poles()[chanIdx]))
		}

		bm, err := cf.BM()
		if err != nil {
			log.Panic(err)
		}
		for chanIdx := 0; chanIdx < cf.NumChannels(); chanIdx++ {
			channel := make([]float64, fftWindowSize)
			for sampleIdx := range channel {
				channel[sampleIdx] = float64(bm[(cf.NumSamples()-fftWindowSize+sampleIdx)*cf.NumChannels()+chanIdx])
			}
			evaluation.Analysis.BMChannels = append(evaluation.Analysis.BMChannels, channel)
			evaluation.Analysis.BMChannelSpectrums = append(evaluation.Analysis.BMChannelSpectrums, spectrum.Compute(channel, rate))
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
		bar.Increment()
	}
	bar.Finish()
}
