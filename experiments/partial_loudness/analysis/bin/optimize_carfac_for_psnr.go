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

	"github.com/cheggaaa/pb"
	"github.com/google-research/korvapuusti/experiments/partial_loudness/analysis"
	"github.com/google-research/korvapuusti/tools/carfac"
	"github.com/google-research/korvapuusti/tools/spectrum"
	"github.com/google-research/korvapuusti/tools/synthesize/signals"
	"github.com/google-research/korvapuusti/tools/workerpool"
	"gonum.org/v1/gonum/optimize"
)

const (
	rate          = 48000
	fftWindowSize = 2048
)

var (
	sqrt2 = math.Sqrt(2.0)

	// Optimize-time outputs.

	outputDir                  = flag.String("output_dir", filepath.Join(os.Getenv("HOME"), "optimize_carfac_for_psnr"), "Directory to put output files in.")
	lossCalculationOutputRatio = flag.Int("loss_calculation_output_ratio", 100, "How seldom to output data about the best/worst results of a calculation run.")

	// Definition of evaluations.

	evaluationJSONGlob           = flag.String("evaluation_json_glob", "", "Glob to the files containing evaluations.")
	evaluationFullScaleSineLevel = flag.Float64("evaluation_full_scale_sine_level", 100.0, "The calibration for a full scale sine during evaluation.")
	noiseFloor                   = flag.Float64("noise_floor", 35, "Noise floor when evaluations were made.")

	// Optimization settings.

	startX         = flag.String("start_x", "", "Starting values in JSON format.")
	pNorm          = flag.Float64("p_norm", 4, "Power of the norm when calculating the loss.")
	skipOpenLoop   = flag.Bool("skip_open_loop", false, "Whether to skip the second (open, less non-linear) run of each signal sample.")
	usingBM        = flag.Bool("using_bm", false, "Whether to use the basilar membrane output of CARFAC (as opposed to the neural activation pattern output).")
	disabledFields = flag.String("disabled_fields", "", "Comma separated fields to avoid optimizing (leave at start value).")
	noLimits       = flag.Bool("no_limits", false, "Disable the limit loss.")

	// Alternative modes.

	exploreField       = flag.String("explore_field", "", "Change mode to explore this field from min limit to max limit, and generating a plot of the loss across that dimension.")
	exploreFieldPoints = flag.Int("explore_field_points", 100, "The number of points to calculate when exploring a field.")
)

func normalize(x float64, scale [2]float64) float64 {
	return (x - scale[0]) / (scale[1] - scale[0])
}

func denormalize(x float64, scale [2]float64) float64 {
	return x*(scale[1]-scale[0]) + scale[0]
}

type optConfig struct {
	OpenLoop       bool
	UsingNAP       bool
	DisabledFields map[string]bool
	PNorm          float64
	Limits         bool
}

type xValues struct {
	CarfacFullScaleSineLevel float64 `start:"100.0" scale:"70.0,130.0" limits:"-,-"`

	VelocityScale float64 `start:"0.1" scale:"0.02,0.5" limits:"-,-"`
	VOffset       float64 `start:"0.04" scale:"0.0,0.5" limits:"-,-"`
	MinZeta       float64 `start:"0.1" scale:"0.01,0.5" limits:"0.0,-"`
	MaxZeta       float64 `start:"0.35" scale:"0.1,0.5" limits:"-,-"`
	ZeroRatio     float64 `start:"1.4142135623730951" scale:"1.2,3.0" limits:"-,-"`
	ERBPerStep    float64 `start:"0.5" scale:"0.1,1.0" limits:"0.1,-"`
	ERBBreakFreq  float64 `start:"165.3" scale:"100.0,200.0" limits:"-,-"`
	ERBQ          float64 `start:"9.264491981582191" scale:"5.0,15.0" limits:"-,-"`

	StageGain       float64 `start:"2.0" scale:"1.2,8.0" limits:"1.2,-"`
	AGC1Scale0      float64 `start:"1.0" scale:"0.5,3.0" limits:"-,-"`
	AGC1ScaleMul    float64 `start:"1.4142135623730951" scale:"1.2,3.0" limits:"-,-"`
	AGC2Scale0      float64 `start:"1.65" scale:"0.2,2.0" limits:"-,-"`
	AGC2ScaleMul    float64 `start:"1.4142135623730951" scale:"1.2,3.0" limits:"-,-"`
	TimeConstant0   float64 `start:"0.002" scale:"0.001,0.008" limits:"-,-"`
	TimeConstantMul float64 `start:"4" scale:"2.0,8.0" limits:"-,-"`

	LoudnessConstant float64 `start:"40.0" scale:"0.0,80.0" limits:"-,-"`
	LoudnessScale    float64 `start:"2.0" scale:"0.1,10.0" limits:"-,-"`
}

func (x xValues) activeValues(conf optConfig) string {
	b, err := json.Marshal(x)
	if err != nil {
		log.Panic(err)
	}
	m := map[string]interface{}{}
	if err := json.Unmarshal(b, &m); err != nil {
		log.Panic(err)
	}
	activeFields := map[string]bool{}
	typ := reflect.TypeOf(x)
	for _, idx := range x.activeFieldIndices(conf) {
		activeFields[typ.Field(idx).Name] = true
	}
	for k := range m {
		if !activeFields[k] {
			delete(m, k)
		}
	}
	b, err = json.Marshal(m)
	if err != nil {
		log.Panic(err)
	}
	return string(b)
}

func (x xValues) limitsForField(idx int) []*float64 {
	typ := reflect.TypeOf(x)
	limits := typ.Field(idx).Tag.Get("limits")
	if limits == "" {
		log.Fatalf("Field %+v doesn't have a limits tag!", typ.Field(idx))
	}
	parts := strings.Split(limits, ",")
	if len(parts) != 2 {
		log.Fatalf("Field %+v doesn't have a limits tag that consists of two comma separated values!", typ.Field(idx))
	}
	result := make([]*float64, 2)
	if parts[0] != "-" {
		lowerLimit, err := strconv.ParseFloat(parts[0], 64)
		if err != nil {
			log.Fatalf("Field %+v has a limits tag whose lower limit isn't parseable as float64: %v", typ.Field(idx), err)
		}
		result[0] = &lowerLimit
	}
	if parts[1] != "-" {
		upperLimit, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			log.Fatalf("Field %+v has a limits tag whose upper limit isn't parseable as float64: %v", typ.Field(idx), err)
		}
		result[1] = &upperLimit
	}
	return result
}

func (x xValues) limitLoss() (float64, []string) {
	val := reflect.ValueOf(x)
	typ := reflect.TypeOf(x)
	result := 0.0
	explanation := []string{}
	for idx := 0; idx < val.NumField(); idx++ {
		current := val.Field(idx).Float()
		normCurrent := normalize(current, x.scaleForField(idx)) * 10
		limits := x.limitsForField(idx)
		if limits[0] != nil {
			if current < *limits[0] {
				normLowerLimit := normalize(*limits[0], x.scaleForField(idx)) * 10
				contribution := math.Pow(normLowerLimit-normCurrent, 2)
				result += contribution
				explanation = append(explanation, fmt.Sprintf("%v = %v (< %v): %v", typ.Field(idx).Name, current, *limits[0], contribution))
			}
		}
		if limits[1] != nil {
			if current > *limits[1] {
				normUpperLimit := normalize(*limits[1], x.scaleForField(idx)) * 10
				contribution := math.Pow(normUpperLimit-normCurrent, 2)
				result += contribution
				explanation = append(explanation, fmt.Sprintf("%v = %v (> %v): %v", typ.Field(idx).Name, current, *limits[1], contribution))
			}
		}
	}
	return result, explanation
}

func (x xValues) activeFieldIndices(conf optConfig) []int {
	typ := reflect.TypeOf(x)
	result := []int{}
	for idx := 0; idx < typ.NumField(); idx++ {
		if !conf.DisabledFields[typ.Field(idx).Name] && (conf.UsingNAP || typ.Field(idx).Tag.Get("nap") != "true") {
			result = append(result, idx)
		}
	}
	return result
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

func (x *xValues) init() {
	val := reflect.ValueOf(x)
	typ := reflect.TypeOf(*x)
	for idx := 0; idx < typ.NumField(); idx++ {
		start, err := strconv.ParseFloat(typ.Field(idx).Tag.Get("start"), 64)
		if err != nil {
			log.Panicf("Field %+v doesn't have a start tag!", typ.Field(idx))
		}
		val.Elem().Field(idx).Set(reflect.ValueOf(start))
	}
}

func (x xValues) toNormalizedFloat64Slice(conf optConfig) []float64 {
	val := reflect.ValueOf(x)
	result := []float64{}
	for _, idx := range x.activeFieldIndices(conf) {
		result = append(result, normalize(val.Field(idx).Float(), x.scaleForField(idx)))
	}
	return result
}

func (x *xValues) setFromNormalizedFloat64Slice(conf optConfig, xSlice []float64) {
	x.init()
	val := reflect.ValueOf(x)
	for _, idx := range x.activeFieldIndices(conf) {
		val.Elem().Field(idx).Set(reflect.ValueOf(denormalize(xSlice[0], x.scaleForField(idx))))
		xSlice = xSlice[1:]
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
	outDir                       string
	evaluationFullScaleSineLevel signals.DB
	lossCalculationOutputRatio   int
	conf                         optConfig

	evaluations      []evaluation
	err              error
	lossCalculations int
}

func (l *lossCalculator) loadEvaluations(glob string, noiseFloor signals.DB) error {
	l.evaluations = nil

	wp := workerpool.New(map[string]int{"L": runtime.NumCPU(), "P": 0, "C": 0})

	evaluationChannel := make(chan evaluation)
	wp.Queue("C", func() error {
		for evaluation := range evaluationChannel {
			l.evaluations = append(l.evaluations, evaluation)
		}
		return nil
	})

	wp.Queue("P", func() error {
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
					wp.Queue("L", func() error {
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
					})
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
		wp.Close("L")
		return nil
	})
	wp.Close("P")
	if err := wp.Wait("L", "P"); err != nil {
		return err
	}
	close(evaluationChannel)
	wp.Close("C")
	return wp.WaitAll()
}

func (l *lossCalculator) loss(x []float64) float64 {
	return l.lossHelper(x, "", "")
}

func (l *lossCalculator) lossHelper(x []float64, forceLogWorstTo string, forceLogAllTo string) float64 {
	l.lossCalculations++
	xValues := &xValues{}
	xValues.setFromNormalizedFloat64Slice(l.conf, x)
	fmt.Printf("Evaluation ** %v ** using %+v with %s\n", l.lossCalculations, l.conf, xValues.activeValues(l.conf))
	carfacParams := carfac.CARFACParams{
		SampleRate: rate,

		VelocityScale: &xValues.VelocityScale,
		VOffset:       &xValues.VOffset,
		MinZeta:       &xValues.MinZeta,
		MaxZeta:       &xValues.MaxZeta,
		ZeroRatio:     &xValues.ZeroRatio,
		ERBPerStep:    &xValues.ERBPerStep,
		ERBBreakFreq:  &xValues.ERBBreakFreq,
		ERBQ:          &xValues.ERBQ,

		StageGain:       &xValues.StageGain,
		AGC1Scale0:      &xValues.AGC1Scale0,
		AGC1ScaleMul:    &xValues.AGC1ScaleMul,
		AGC2Scale0:      &xValues.AGC2Scale0,
		AGC2ScaleMul:    &xValues.AGC2ScaleMul,
		TimeConstant0:   &xValues.TimeConstant0,
		TimeConstantMul: &xValues.TimeConstantMul,
	}

	carfacs := make(chan carfac.CF, runtime.NumCPU())
	for i := 0; i < runtime.NumCPU(); i++ {
		carfacs <- carfac.New(carfacParams)
	}

	bar := pb.StartNew(len(l.evaluations)).Prefix("Evaluating")

	psnrChan := make(chan psnr, len(l.evaluations))
	wp := workerpool.New(map[string]int{"L": runtime.NumCPU(), "P": 0})
	wp.Queue("P", func() error {
		for _, evaluationVar := range l.evaluations {
			evaluation := evaluationVar
			wp.Queue("L", func() error {
				cf := <-carfacs
				defer func() { carfacs <- cf }()
				scaledSignal := evaluation.signal.ToFloat32AddLevel(l.evaluationFullScaleSineLevel - signals.DB(xValues.CarfacFullScaleSineLevel))
				cf.Reset()
				cf.Run(scaledSignal[len(evaluation.signal)-cf.NumSamples():])
				if l.conf.OpenLoop {
					cf.RunOpen(scaledSignal[len(evaluation.signal)-cf.NumSamples():])
				}
				var cfOut []float32
				var err error
				if !l.conf.UsingNAP {
					cfOut, err = cf.BM()
				} else {
					cfOut, err = cf.NAP()
				}
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
						channel[sampleIdx] = float64(cfOut[(cf.NumSamples()-fftWindowSize+sampleIdx)*cf.NumChannels()+chanIdx])
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
			})
		}
		wp.Close("L")
		return nil
	})
	wp.Close("P")
	if err := wp.WaitAll(); err != nil {
		l.err = err
		return 0.0
	}
	close(psnrChan)
	bar.Finish()
	sumOfSquares := 0.0
	psnrsByRunID := map[string]psnrs{}
	lossByRunID := map[string]float64{}
	for evalPSNR := range psnrChan {
		psnrsByRunID[evalPSNR.evaluation.runID] = append(psnrsByRunID[evalPSNR.evaluation.runID], evalPSNR)
		predictedLoudnessError := evalPSNR.predictedLoudness - evalPSNR.evaluation.evaluatedLoudness
		if evalPSNR.evaluation.evaluatedLoudness < 27 {
			errorDiscount := signals.DB(30-evalPSNR.evaluation.evaluatedLoudness) / 3
			predictedLoudnessError /= errorDiscount
		}
		square := math.Pow(float64(predictedLoudnessError*predictedLoudnessError), l.conf.PNorm)
		sumOfSquares += square
		lossByRunID[evalPSNR.evaluation.runID] += square
	}
	worstRun := psnrs{}
	worstAvgLoss := 0.0
	for runID := range lossByRunID {
		lossByRunID[runID] = math.Pow(lossByRunID[runID]/float64(len(psnrsByRunID[runID])), 1.0/l.conf.PNorm)
		if lossByRunID[runID] > worstAvgLoss {
			worstAvgLoss = lossByRunID[runID]
			worstRun = psnrsByRunID[runID]
		}
	}
	if forceLogWorstTo != "" || l.lossCalculations%l.lossCalculationOutputRatio == 0 {
		name := "worst_evaluation_run"
		if forceLogWorstTo != "" {
			name = forceLogWorstTo
		}
		if err := l.logPSNRs(worstRun, name); err != nil {
			l.err = err
			return 0.0
		}
	}
	if forceLogAllTo != "" {
		all := psnrs{}
		for _, runs := range psnrsByRunID {
			all = append(all, runs...)
		}
		if err := l.logPSNRs(all, forceLogAllTo); err != nil {
			l.err = err
			return 0.0
		}
	}
	loss := math.Pow(sumOfSquares/float64(len(l.evaluations)), 1.0/l.conf.PNorm)
	limitLoss := 0.0
	explanation := []string{"limit loss disabled"}
	if l.conf.Limits {
		limitLoss, explanation = xValues.limitLoss()
	}
	totalLoss := loss + limitLoss
	fmt.Printf("Got loss %v (limit loss %v: %v)\n", totalLoss, limitLoss, strings.Join(explanation, ", "))
	return totalLoss
}

type plot struct {
	x     []float64
	y     []float64
	label string
}

func (l *lossCalculator) makePythonPlot(filename string, title string, xlabel string, ylabel string, plots []plot) error {
	plotCommands := []string{}
	for _, p := range plots {
		marshalledXValues, err := json.Marshal(p.x)
		if err != nil {
			return err
		}
		marshalledYValues, err := json.Marshal(p.y)
		if err != nil {
			return err
		}
		plotCommands = append(plotCommands, fmt.Sprintf("ax.plot(%s, %s, label='%s')", marshalledXValues, marshalledYValues, p.label))
	}
	if err := ioutil.WriteFile(filename, []byte(fmt.Sprintf(`#!/usr/bin/python3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot()
ax.set_title('%s')
ax.set_xlabel('%s')
ax.set_ylabel('%s')
%s
plt.legend()
plt.show()
`, title, xlabel, ylabel, strings.Join(plotCommands, "\n"))), 0777); err != nil {
		return err
	}
	return nil
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
	if err := l.makePythonPlot(logFile, fmt.Sprintf("Optimization run %v/Evaluation run %v", l.lossCalculations, p[0].evaluation.runID), "Evaluation index", "Loudness", []plot{
		{
			x:     xValues,
			y:     evaluatedYValues,
			label: "Evaluated",
		},
		{
			x:     xValues,
			y:     predictedYValues,
			label: "Predicted",
		},
	}); err != nil {
		return err
	}
	fmt.Printf("Saved psnr plot to %v\n", logFile)
	return nil
}

func test() {
	for _, usingNAP := range []bool{true, false} {
		conf := optConfig{UsingNAP: usingNAP}
		testXValues := xValues{}
		xSlice := testXValues.toNormalizedFloat64Slice(conf)
		for idx := range xSlice {
			xSlice[idx] = float64(idx)
		}
		preRoundtripValues := &xValues{}
		preRoundtripValues.setFromNormalizedFloat64Slice(conf, xSlice)
		postRoundtripValues := preRoundtripValues.toNormalizedFloat64Slice(conf)
		if len(xSlice) != len(postRoundtripValues) {
			log.Panicf("Converting back and forth between xValues doesn't provide the same result! Got %+v, wanted %+v", postRoundtripValues, xSlice)
		}
		for idx := range xSlice {
			if math.Abs(xSlice[idx]-postRoundtripValues[idx]) > 0.0000000001 {
				log.Panicf("Converting back and forth between xValues doesn't provide the same result! Got %+v, wanted %+v", postRoundtripValues, xSlice)
			}
		}
	}
}

func (l *lossCalculator) optimize() {
	problem := optimize.Problem{
		Func: l.loss,
		Status: func() (optimize.Status, error) {
			return optimize.NotTerminated, l.err
		},
	}
	initX := &xValues{}
	if *startX == "" {
		initX.init()
	} else {
		if err := json.Unmarshal([]byte(*startX), initX); err != nil {
			log.Fatal(err)
		}
	}
	res, err := optimize.Minimize(problem, initX.toNormalizedFloat64Slice(l.conf), nil, nil)
	if err != nil {
		log.Fatal(err)
	}

	resultValues := &xValues{}
	resultValues.setFromNormalizedFloat64Slice(l.conf, res.Location.X)
	finalFile, err := os.Create(filepath.Join(*outputDir, "final_results.json"))
	if err != nil {
		log.Fatal(err)
	}
	defer finalFile.Close()
	if err := json.NewEncoder(finalFile).Encode(map[string]interface{}{
		"X":    resultValues,
		"Conf": l.conf,
		"Loss": l.lossHelper(resultValues.toNormalizedFloat64Slice(l.conf), "worst_evaluation_run_final_results", "all_evaluation_runs_final_results"),
	}); err != nil {
		log.Fatal(err)
	}

	b, err := json.Marshal(resultValues)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Final results: %+v\n%v\n%s\n", res.Stats, res.Status, b)
}

func (l *lossCalculator) explore(field string, points int) {
	initX := &xValues{}
	initX.init()
	initXTyp := reflect.TypeOf(*initX)
	fieldIdx := -1
	for idx := 0; idx < initXTyp.NumField(); idx++ {
		if initXTyp.Field(idx).Name == field {
			fieldIdx = idx
			break
		}
	}
	if fieldIdx == -1 {
		log.Panicf("Unknown field %q!", field)
	}
	scale := initX.scaleForField(fieldIdx)
	step := (scale[1] - scale[0]) / float64(points)
	vals := []float64{}
	losses := []float64{}
	initXVal := reflect.ValueOf(initX)
	for val := scale[0]; val < scale[1]; val += step {
		vals = append(vals, val)
		initXVal.Elem().Field(fieldIdx).Set(reflect.ValueOf(val))
		loss := l.loss(initX.toNormalizedFloat64Slice(l.conf))
		losses = append(losses, loss)
	}
	logFile := filepath.Join(l.outDir, fmt.Sprintf("exploration_of_%s_over_%v_points.py", field, points))
	if err := l.makePythonPlot(logFile, fmt.Sprintf("Exploration of loss over %s", field), field, "Loss", []plot{{x: vals, y: losses, label: "Loss"}}); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Plotted exploration of %v over %v points to %v\n", field, points, logFile)
}

func main() {
	test()

	flag.Parse()
	if *evaluationJSONGlob == "" {
		flag.Usage()
		os.Exit(1)
	}
	lc := &lossCalculator{
		outDir:                       *outputDir,
		evaluationFullScaleSineLevel: signals.DB(*evaluationFullScaleSineLevel),
		lossCalculationOutputRatio:   *lossCalculationOutputRatio,
		conf: optConfig{
			PNorm:          *pNorm,
			OpenLoop:       !*skipOpenLoop,
			UsingNAP:       !*usingBM,
			DisabledFields: map[string]bool{},
			Limits:         !*noLimits,
		},
	}
	for _, disabledField := range strings.Split(*disabledFields, ",") {
		if strings.TrimSpace(disabledField) != "" {
			lc.conf.DisabledFields[disabledField] = true
		}
	}
	if err := lc.loadEvaluations(*evaluationJSONGlob, signals.DB(*noiseFloor-*evaluationFullScaleSineLevel)); err != nil {
		log.Fatal(err)
	}
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatal(err)
	}

	if *exploreField != "" {
		lc.explore(*exploreField, *exploreFieldPoints)
	} else {
		lc.optimize()
	}
}
