/* optimize_carfac_for_psnr runs CARFAC on the audio used in evaluations, calculate the
 * PSNR across channels and frequencies for the probe, and outputs the error in a prediction
 * of the partial loudness of the probe.
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
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/cheggaaa/pb"
	"github.com/google-research/korvapuusti/experiments/partial_loudness/analysis"
	"github.com/google-research/korvapuusti/tools/carfac"
	"github.com/google-research/korvapuusti/tools/spectrum"
	"github.com/google-research/korvapuusti/tools/synthesize/signals"
	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/gonum/stat"
)

const (
	rate          = 48000
	fftWindowSize = 2048
)

var (
	evaluationJSONGlob           = flag.String("evaluation_json_glob", "", "Glob to the files containing evaluations.")
	evaluationFullScaleSineLevel = flag.Float64("evaluation_full_scale_sine_level", 100, "dB SPL calibrated to a full scale sine in the evaluations.")
	noiseFloor                   = flag.Float64("noise_floor", 35, "dB SPL of noise where evaluations were made.")
)

type multiErr []error

func (m multiErr) Error() string {
	return fmt.Sprint([]error(m))
}

func startWorkerPool() (prioJobs chan func() error, ticketJobs chan func() error, errorPromise func() error) {
	ticketJobs = make(chan func() error)
	prioJobs = make(chan func() error)
	errors := make(chan error)

	go func() {
		defer close(errors)
		wg := &sync.WaitGroup{}
		wg.Add(1)
		go func() {
			defer wg.Done()
			tickets := make(chan struct{}, runtime.NumCPU())
			for job := range ticketJobs {
				tickets <- struct{}{}
				wg.Add(1)
				go func(job func() error) {
					defer wg.Done()
					defer func() { <-tickets }()
					errors <- job()
				}(job)
			}
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range prioJobs {
				wg.Add(1)
				go func(job func() error) {
					defer wg.Done()
					errors <- job()
				}(job)
			}
		}()
		wg.Wait()
	}()
	return prioJobs, ticketJobs, func() error {
		me := multiErr{}
		for err := range errors {
			if err != nil {
				me = append(me, err)
			}
		}
		if len(me) == 0 {
			return nil
		}
		return me
	}
}

type evaluation struct {
	signal            []float32
	probeSampler      signals.Noise
	evaluatedLoudness signals.DB
}

type psnr struct {
	evaluation evaluation
	psnr       float64
}

type lossCalculator struct {
	evaluations []evaluation
	err         error
}

func (l *lossCalculator) loadEvaluations(glob string, noiseFloor signals.DB) error {
	l.evaluations = nil
	evaluationChannel := make(chan evaluation)
	defer close(evaluationChannel)
	go func() {
		for evaluation := range evaluationChannel {
			l.evaluations = append(l.evaluations, evaluation)
		}
	}()

	prioJobs, ticketJobs, errorPromise := startWorkerPool()

	prioJobs <- func() error {
		evaluationFileNames, err := filepath.Glob(glob)
		if err != nil {
			return err
		}

		allFilesBar := pb.New(len(evaluationFileNames)).Prefix("Loading evaluation files")
		fileBar := pb.New(0)
		barPool, err := pb.StartPool(allFilesBar, fileBar)
		if err != nil {
			return err
		}

		for _, evaluationFileName := range evaluationFileNames {
			fileBar.Prefix(evaluationFileName)
			evaluationFile, err := os.Open(evaluationFileName)
			if err != nil {
				return err
			}
			if err := func() error {
				defer evaluationFile.Close()
				lineReader := bufio.NewReader(evaluationFile)
				lines := [][]byte{}
				for line, err := lineReader.ReadString('\n'); err == nil; line, err = lineReader.ReadString('\n') {
					lines = append(lines, []byte(line))
				}
				if err != nil && err != io.EOF {
					log.Panic(err)
				}
				fileBar.SetTotal(len(lines))
				fileBar.Set(0)
				for _, lineVar := range lines {
					line := lineVar
					ticketJobs <- func() error {
						equivalentLoudness := &analysis.EquivalentLoudness{}
						if err := json.Unmarshal(line, equivalentLoudness); err != nil {
							return err
						}
						if equivalentLoudness.EntryType == "EquivalentLoudnessMeasurement" {
							eval := evaluation{
								evaluatedLoudness: signals.DB(equivalentLoudness.Results.ProbeDBSPLForEquivalentLoudness),
							}
							probeSampler, err := equivalentLoudness.Evaluation.Probe.Sampler()
							if err != nil {
								return err
							}
							if noise, ok := probeSampler.(*signals.Noise); ok {
								eval.probeSampler = *noise
							} else {
								return fmt.Errorf("Probe sampler %+v isn't a Noise sampler?", probeSampler)
							}
							combinedSampler, err := equivalentLoudness.Evaluation.Combined.Sampler()
							if err != nil {
								return err
							}
							noisySampler := signals.Superposition{combinedSampler, &signals.Noise{Color: signals.White, LowerLimit: 20, UpperLimit: 20000, Level: noiseFloor}}
							signal, err := noisySampler.Sample(signals.TimeStretch{FromInclusive: 0, ToExclusive: 1}, rate, nil)
							if err != nil {
								return err
							}
							eval.signal = make([]float32, len(signal))
							for idx := range signal {
								eval.signal[idx] = float32(signal[idx])
							}
							evaluationChannel <- eval
						}
						fileBar.Increment()
						return nil
					}
				}
				return nil
			}(); err != nil {
				return err
			}
			fileBar.Finish()
			allFilesBar.Increment()
		}
		allFilesBar.Finish()
		barPool.Stop()
		close(ticketJobs)
		return nil
	}
	close(prioJobs)
	return errorPromise()
}

func (l *lossCalculator) loss(x []float64) float64 {
	mz := x[0]
	zr := x[1]
	sg := x[2]
	carfacParams := carfac.CARFACParams{
		SampleRate: rate,
		MaxZeta:    &mz,
		ZeroRatio:  &zr,
		StageGain:  &sg,
	}
	loudnessConstant := x[3]
	loudnessScale := x[4]
	fmt.Printf("Evaluating with x %+v\n", x)

	carfacs := make(chan carfac.CF, runtime.NumCPU())
	for i := 0; i < runtime.NumCPU(); i++ {
		carfacs <- carfac.New(carfacParams)
	}

	bar := pb.StartNew(len(l.evaluations)).Prefix("Evaluating")

	psnrs := make(chan psnr, len(l.evaluations))
	prioJobs, ticketJobs, errorPromise := startWorkerPool()
	prioJobs <- func() error {
		for _, evaluationVar := range l.evaluations {
			evaluation := evaluationVar
			ticketJobs <- func() error {
				cf := <-carfacs
				defer func() { carfacs <- cf }()
				cf.Run(evaluation.signal[len(evaluation.signal)-cf.NumSamples():])
				bm, err := cf.BM()
				if err != nil {
					return err
				}
				evalPSNR := psnr{
					evaluation: evaluation,
					psnr:       -1,
				}
				for chanIdx := 0; chanIdx < cf.NumChannels(); chanIdx++ {
					channel := make([]float64, fftWindowSize)
					for sampleIdx := range channel {
						channel[sampleIdx] = float64(bm[(cf.NumSamples()-fftWindowSize+sampleIdx)*cf.NumChannels()+chanIdx])
					}
					spec := spectrum.ComputeSignalPower(channel, rate)
					energy := stat.Variance(channel, nil)
					for binIdx := int(math.Floor(float64(evaluation.probeSampler.LowerLimit / spec.BinWidth))); binIdx <= int(math.Ceil(float64(evaluation.probeSampler.UpperLimit/spec.BinWidth))); binIdx++ {
						if snr := 10*math.Log10(spec.SignalPower[binIdx]) - 10*math.Log10(energy); snr > evalPSNR.psnr {
							evalPSNR.psnr = snr
						}
					}
				}
				psnrs <- evalPSNR
				bar.Increment()
				return nil
			}
		}
		close(ticketJobs)
		return nil
	}
	close(prioJobs)
	if err := errorPromise(); err != nil {
		l.err = err
		return 0.0
	}
	close(psnrs)
	bar.Finish()
	sumOfSquares := 0.0
	for evalPSNR := range psnrs {
		predictedLoudness := signals.DB(loudnessConstant + loudnessScale*evalPSNR.psnr)
		predictedLoudnessError := predictedLoudness - evalPSNR.evaluation.evaluatedLoudness
		sumOfSquares += float64(predictedLoudnessError * predictedLoudnessError)
	}
	loss := math.Pow(sumOfSquares/float64(len(l.evaluations)), 0.5)
	fmt.Println("Got loss", loss)
	return loss
}

func main() {
	flag.Parse()
	if *evaluationJSONGlob == "" {
		flag.Usage()
		os.Exit(1)
	}
	lc := &lossCalculator{}
	if err := lc.loadEvaluations(*evaluationJSONGlob, signals.DB(*noiseFloor-*evaluationFullScaleSineLevel)); err != nil {
		log.Fatal(err)
	}
	problem := optimize.Problem{
		Func: lc.loss,
		Status: func() (optimize.Status, error) {
			return optimize.NotTerminated, lc.err
		},
	}
	initX := []float64{0.35, math.Sqrt(2.0), 2.0, 10.0, 9.0}
	res, err := optimize.Minimize(problem, initX, nil, nil)
	fmt.Println(res, err)

}
