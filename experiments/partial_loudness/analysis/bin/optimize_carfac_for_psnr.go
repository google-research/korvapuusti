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
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/cheggaaa/pb"
	"github.com/google-research/korvapuusti/experiments/partial_loudness/analysis"
	"github.com/google-research/korvapuusti/tools/carfac"
	"github.com/google-research/korvapuusti/tools/spectrum"
	"github.com/google-research/korvapuusti/tools/synthesize/signals"
	"gonum.org/v1/gonum/optimize"
)

const (
	rate          = 48000
	fftWindowSize = 2048
)

var (
	// Optimize-time outputs.

	outputDir                  = flag.String("output_dir", filepath.Join(os.Getenv("HOME"), "optimize_carfac_for_psnr"), "Directory to put output files in.")
	lossCalculationOutputRatio = flag.Int("loss_calculation_output_ratio", 100, "How seldom to output data about the best/worst results of a calculation run.")

	// Definition of evaluations.

	evaluationJSONGlob = flag.String("evaluation_json_glob", "", "Glob to the files containing evaluations.")
	noiseFloor         = flag.Float64("noise_floor", 35, "Noise floor when evaluations were made.")
	pNorm              = flag.Float64("p_norm", 2, "Power of the norm when calculating the loss.")

	// Start values for optimization.

	evaluationFullScaleSineLevel  = flag.Float64("evaluation_full_scale_sine_level", 100, "dB SPL calibrated to a full scale sine in the evaluations.")
	carfacFullScaleSineLevelStart = flag.Float64("carfac_full_scale_sine_level_start", 100, "carfac_full_scale_sine_level starting point.")
	maxZetaStart                  = flag.Float64("max_zeta_start", 0.35, "max_zeta starting point.")
	zeroRatioStart                = flag.Float64("zero_ratio_start", math.Sqrt(2.0), "zero_ratio starting point.")
	stageGainStart                = flag.Float64("stage_gain_start", 2.0, "stage_gain starting point.")
	vOffsetStart                  = flag.Float64("v_offset_start", 0.04, "v_offset starting point.")
	loudnessConstantStart         = flag.Float64("loudness_constant_start", 40.0, "Loudness constant starting point.")
	loudnessScaleStart            = flag.Float64("loudness_scale_start", 2, "Loudness scale starting point.")
)

func normalize(x float64, scale [2]float64) float64 {
	return (x - scale[0]) / (scale[1] - scale[0])
}

func denormalize(x float64, scale [2]float64) float64 {
	return x*(scale[1]-scale[0]) + scale[0]
}

type xValues struct {
	CarfacFullScaleSineLevel float64 `scale:"70.0,130.0"`
	MaxZeta                  float64 `scale:"0.1,0.5"`
	ZeroRatio                float64 `scale:"0.5,3.0"`
	StageGain                float64 `scale:"1.0,4.0"`
	VOffset                  float64 `scale:"0.01,0.1"`
	LoudnessConstant         float64 `scale:"0.0,80.0"`
	LoudnessScale            float64 `scale:"0.1,10.0"`
}

func (x xValues) scaleForField(fieldIdx int) [2]float64 {
	parts := strings.Split(reflect.TypeOf(x).Field(fieldIdx).Tag.Get("scale"), ",")
	min, err := strconv.ParseFloat(parts[0], 64)
	if err != nil {
		panic(err)
	}
	max, err := strconv.ParseFloat(parts[1], 64)
	if err != nil {
		panic(err)
	}
	return [2]float64{min, max}
}

func initXValues() xValues {
	return xValues{
		CarfacFullScaleSineLevel: *carfacFullScaleSineLevelStart,
		MaxZeta:                  *maxZetaStart,
		ZeroRatio:                *zeroRatioStart,
		StageGain:                *stageGainStart,
		VOffset:                  *vOffsetStart,
		LoudnessConstant:         *loudnessConstantStart,
		LoudnessScale:            *loudnessScaleStart,
	}
}

func (x xValues) toNormalizedFloat64Slice() []float64 {
	val := reflect.ValueOf(x)
	result := make([]float64, val.NumField())
	for idx := range result {
		result[idx] = normalize(val.Field(idx).Float(), x.scaleForField(idx))
	}
	return result
}

func xValuesFromNormalizedFloat64Slice(x []float64) xValues {
	result := &xValues{}
	val := reflect.ValueOf(result)
	for idx := range x {
		val.Elem().Field(idx).Set(reflect.ValueOf(denormalize(x[idx], result.scaleForField(idx))))
	}
	return *result
}

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
	signal            signals.Float64Slice
	evaluationID      string
	runID             string
	probeSampler      signals.Noise
	evaluatedLoudness signals.DB
}

type psnr struct {
	evaluation        evaluation
	psnr              float64
	predictedLoudness signals.DB
}

type psnrs []psnr

func (p psnrs) Len() int {
	return len(p)
}

func (p psnrs) Less(i, j int) bool {
	iParts := strings.Split(p[i].evaluation.evaluationID, "_")
	jParts := strings.Split(p[j].evaluation.evaluationID, "_")
	iTime, err := strconv.ParseInt(iParts[0], 10, 64)
	if err != nil {
		panic(err)
	}
	jTime, err := strconv.ParseInt(jParts[0], 10, 64)
	if err != nil {
		panic(err)
	}
	return iTime < jTime
}

func (p psnrs) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

type lossCalculator struct {
	evaluations                  []evaluation
	err                          error
	outDir                       string
	lossCalculations             int
	lossCalculationOutputRatio   int
	pNorm                        float64
	evaluationFullScaleSineLevel signals.DB
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
								evaluationID:      equivalentLoudness.Evaluation.ID,
								runID:             equivalentLoudness.Run.ID,
							}
							probeSampler, err := equivalentLoudness.Evaluation.Probe.Sampler()
							if err != nil {
								return err
							}
							probeNoise, ok := probeSampler.(*signals.Noise)
							if !ok {
								return fmt.Errorf("Probe sampler %+v isn't a Noise sampler?", probeSampler)
							}
							eval.probeSampler = *probeNoise
							combinedSampler, err := equivalentLoudness.Evaluation.Combined.Sampler()
							if err != nil {
								return err
							}
							superpos, ok := combinedSampler.(signals.Superposition)
							if !ok {
								return fmt.Errorf("Combined sampler %+v isn't a Superposition sampler?", combinedSampler)
							}
							for _, sampler := range superpos {
								noise, ok := sampler.(*signals.Noise)
								if !ok {
									return fmt.Errorf("Combined part sampler %+v isn't a Noise sampler?", sampler)
								}
								if noise.LowerLimit == probeNoise.LowerLimit && noise.UpperLimit == probeNoise.UpperLimit {
									continue
								}
								if noise.LowerLimit <= probeNoise.LowerLimit && noise.UpperLimit >= probeNoise.UpperLimit {
									return nil
								}
								if noise.LowerLimit >= probeNoise.LowerLimit && noise.LowerLimit <= probeNoise.UpperLimit {
									return nil
								}
								if noise.UpperLimit >= probeNoise.LowerLimit && noise.UpperLimit <= probeNoise.UpperLimit {
									return nil
								}
							}
							noisySampler := signals.Superposition{combinedSampler, &signals.Noise{Color: signals.White, LowerLimit: 20, UpperLimit: 20000, Level: noiseFloor}}
							eval.signal, err = noisySampler.Sample(signals.TimeStretch{FromInclusive: 0, ToExclusive: 1}, rate, nil)
							if err != nil {
								return err
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
	l.lossCalculations++
	xValues := xValuesFromNormalizedFloat64Slice(x)
	carfacParams := carfac.CARFACParams{
		SampleRate: rate,
		MaxZeta:    &xValues.MaxZeta,
		ZeroRatio:  &xValues.ZeroRatio,
		StageGain:  &xValues.StageGain,
		VOffset:    &xValues.VOffset,
	}
	fmt.Printf("Evaluating with %+v\n", xValues)

	carfacs := make(chan carfac.CF, runtime.NumCPU())
	for i := 0; i < runtime.NumCPU(); i++ {
		carfacs <- carfac.New(carfacParams)
	}

	bar := pb.StartNew(len(l.evaluations)).Prefix("Evaluating")

	psnrChan := make(chan psnr, len(l.evaluations))
	prioJobs, ticketJobs, errorPromise := startWorkerPool()
	prioJobs <- func() error {
		for _, evaluationVar := range l.evaluations {
			evaluation := evaluationVar
			ticketJobs <- func() error {
				cf := <-carfacs
				defer func() { carfacs <- cf }()
				scaledSignal := evaluation.signal.ToFloat32AddLevel(l.evaluationFullScaleSineLevel - signals.DB(xValues.CarfacFullScaleSineLevel))
				cf.Run(scaledSignal[len(evaluation.signal)-cf.NumSamples():])
				bm, err := cf.BM()
				if err != nil {
					return err
				}
				evalPSNR := psnr{
					evaluation: evaluation,
					psnr:       -10000,
				}
				for chanIdx := 0; chanIdx < cf.NumChannels(); chanIdx++ {
					channel := make([]float64, fftWindowSize)
					for sampleIdx := range channel {
						channel[sampleIdx] = float64(bm[(cf.NumSamples()-fftWindowSize+sampleIdx)*cf.NumChannels()+chanIdx])
					}
					spec := spectrum.Compute(channel, rate)
					for binIdx := int(math.Floor(float64(evaluation.probeSampler.LowerLimit / spec.BinWidth))); binIdx <= int(math.Ceil(float64(evaluation.probeSampler.UpperLimit/spec.BinWidth))); binIdx++ {
						snr := spec.SignalPower[binIdx] - spec.NoisePower[binIdx]
						if snr > evalPSNR.psnr {
							evalPSNR.psnr = snr
						}
					}
				}
				evalPSNR.predictedLoudness = signals.DB(xValues.LoudnessConstant + xValues.LoudnessScale*evalPSNR.psnr)
				psnrChan <- evalPSNR
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
	close(psnrChan)
	bar.Finish()
	sumOfSquares := 0.0
	var worstEval *psnr
	worstSquare := 0.0
	psnrByRunID := map[string]psnrs{}
	for evalPSNR := range psnrChan {
		psnrByRunID[evalPSNR.evaluation.runID] = append(psnrByRunID[evalPSNR.evaluation.runID], evalPSNR)
		predictedLoudnessError := evalPSNR.predictedLoudness - evalPSNR.evaluation.evaluatedLoudness
		square := math.Pow(float64(predictedLoudnessError*predictedLoudnessError), l.pNorm)
		if square > worstSquare {
			worstSquare = square
			worstEval = &evalPSNR
		}
		sumOfSquares += square
	}
	if l.lossCalculations%l.lossCalculationOutputRatio == 0 {
		if err := l.logPSNRs(psnrByRunID[worstEval.evaluation.runID], "worst_evaluation_run"); err != nil {
			l.err = err
			return 0.0
		}
	}
	loss := math.Pow(sumOfSquares/float64(len(l.evaluations)), 1.0/l.pNorm)
	fmt.Println("Got loss", loss)
	return loss
}

func (l *lossCalculator) logPSNRs(p psnrs, name string) error {
	if err := os.MkdirAll(l.outDir, 0777); err != nil {
		return err
	}
	logFile := filepath.Join(l.outDir, fmt.Sprintf("%v_for_optimization_run_%v.py", name, l.lossCalculations))
	xValues := []float64{}
	evaluatedYValues := []float64{}
	predictedYValues := []float64{}
	sort.Sort(p)
	for idx := range p {
		xValues = append(xValues, float64(idx))
		evaluatedYValues = append(evaluatedYValues, float64(p[idx].evaluation.evaluatedLoudness))
		predictedYValues = append(predictedYValues, float64(p[idx].predictedLoudness))
	}
	marshalledXValues, err := json.Marshal(xValues)
	if err != nil {
		return err
	}
	marshalledEvaluatedYValues, err := json.Marshal(evaluatedYValues)
	if err != nil {
		return err
	}
	marshalledPredictedYValues, err := json.Marshal(predictedYValues)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(logFile, []byte(fmt.Sprintf(`#!/usr/bin/python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot()
ax.set_title('Optimization run %v/Evaluation run %v')
ax.set_xlabel('Evaluation index')
ax.set_ylabel('Loudness')
ax.plot(%s, %s, label='Evaluated')
ax.plot(%s, %s, label='Predicted')
plt.legend()
plt.show()
`, l.lossCalculations, p[0].evaluation.runID, marshalledXValues, marshalledEvaluatedYValues, marshalledXValues, marshalledPredictedYValues)), 0777)
}

func main() {
	flag.Parse()
	if *evaluationJSONGlob == "" {
		flag.Usage()
		os.Exit(1)
	}
	lc := &lossCalculator{
		outDir:                       *outputDir,
		evaluationFullScaleSineLevel: signals.DB(*evaluationFullScaleSineLevel),
		lossCalculationOutputRatio:   *lossCalculationOutputRatio,
	}
	if err := lc.loadEvaluations(*evaluationJSONGlob, signals.DB(*noiseFloor-*evaluationFullScaleSineLevel)); err != nil {
		log.Fatal(err)
	}
	problem := optimize.Problem{
		Func: lc.loss,
		Status: func() (optimize.Status, error) {
			return optimize.NotTerminated, lc.err
		},
	}
	initX := initXValues().toNormalizedFloat64Slice()
	res, err := optimize.Minimize(problem, initX, nil, nil)
	fmt.Println(res, err)

}
