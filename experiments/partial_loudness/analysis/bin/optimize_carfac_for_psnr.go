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
	"bytes"
	"crypto/sha1"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"net/http"
	"net/rpc"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/cheggaaa/pb"
	"github.com/google-research/korvapuusti/experiments/partial_loudness/analysis"
	"github.com/google-research/korvapuusti/tools/carfac"
	"github.com/google-research/korvapuusti/tools/spectrum"
	"github.com/google-research/korvapuusti/tools/synthesize/signals"
	"github.com/google-research/korvapuusti/tools/workerpool"
	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/gonum/stat"
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
	remoteComputers            = flag.String("remote_computers", "", "Comma separated list of host:port pairs defining remote computers to use. Not providing this will instead run a remote computer.")

	// Definition of evaluations.

	evaluationJSONGlob           = flag.String("evaluation_json_glob", "", "Glob to the files containing evaluations.")
	evaluationFullScaleSineLevel = flag.Float64("evaluation_full_scale_sine_level", 100.0, "The calibration for a full scale sine during evaluation.")
	noiseFloor                   = flag.Float64("noise_floor", 35, "Noise floor when evaluations were made.")

	// Optimization settings.

	startX         = flag.String("start_x", "", "Starting values in JSON format.")
	pNorm          = flag.Float64("p_norm", 4, "Power of the norm when calculating the loss.")
	skipOpenLoop   = flag.Bool("skip_open_loop", false, "Whether to skip the second (open, less non-linear) run of each signal sample.")
	usingBM        = flag.Bool("using_bm", true, "Whether to use the basilar membrane output of CARFAC (as opposed to the neural activation pattern output).")
	disabledFields = flag.String("disabled_fields", "", "Comma separated fields to avoid optimizing (leave at start value).")
	noLimits       = flag.Bool("no_limits", false, "Disable the limit loss.")
	useSNNR        = flag.Bool("use_snnr", true, "Use SNNR instead of SNR to estimate partial loudness.")
	erbPerStep     = flag.Float64("erb_per_step", 0.25, "erb_per_step while running CARFAC.")

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

type OptConfig struct {
	OpenLoop                     bool
	UsingNAP                     bool
	DisabledFields               map[string]bool
	PNorm                        float64
	Limits                       bool
	UseSNNR                      bool
	ERBPerStep                   float64
	EvaluationFullScaleSineLevel signals.DB
	NoiseFloor                   signals.DB
}

func (o *OptConfig) zeta(zetaConst float64) float64 {
	// Based on the assumtion that max small-signal gain at the passband peak
	// will be on the order of  (0.5/min_zeta)^(1/erb_per_step), and we need
	// the start value of that in the same region or the loss function becomes
	// too uneven to optimize.
	return math.Pow(zetaConst, -o.ERBPerStep) * math.Pow(math.Pow(0.5, -1.0/o.ERBPerStep), -o.ERBPerStep)
}

type XValues struct {
	CarfacFullScaleSineLevel float64 `start:"100.0" scale:"70.0,130.0" limits:"-,-"`

	VelocityScale float64 `start:"0.1" scale:"0.02,0.5" limits:"-,-"`
	VOffset       float64 `start:"0.04" scale:"0.0,0.5" limits:"-,-"`
	MinZeta       float64 `scale:"0.01,0.5" limits:"0.0,-"`
	MaxZeta       float64 `scale:"0.1,5.0" limits:"-,-"`
	ZeroRatio     float64 `start:"1.4142135623730951" scale:"1.2,3.0" limits:"-,-"`

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

func (x *XValues) activeValues(conf *OptConfig) string {
	b, err := json.Marshal(x)
	if err != nil {
		log.Panic(err)
	}
	m := map[string]interface{}{}
	if err := json.Unmarshal(b, &m); err != nil {
		log.Panic(err)
	}
	activeFields := map[string]bool{}
	typ := reflect.TypeOf(*x)
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

func (x *XValues) limitsForField(idx int) []*float64 {
	typ := reflect.TypeOf(*x)
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

func (x *XValues) limitLoss() (float64, []string) {
	val := reflect.ValueOf(*x)
	typ := reflect.TypeOf(*x)
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

func (x *XValues) activeFieldIndices(conf *OptConfig) []int {
	typ := reflect.TypeOf(*x)
	result := []int{}
	for idx := 0; idx < typ.NumField(); idx++ {
		if !conf.DisabledFields[typ.Field(idx).Name] && (conf.UsingNAP || typ.Field(idx).Tag.Get("nap") != "true") {
			result = append(result, idx)
		}
	}
	return result
}

func (x *XValues) scaleForField(fieldIdx int) [2]float64 {
	parts := strings.Split(reflect.TypeOf(*x).Field(fieldIdx).Tag.Get("scale"), ",")
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

func (x *XValues) init(conf *OptConfig) {
	val := reflect.ValueOf(x)
	typ := reflect.TypeOf(*x)
	for idx := 0; idx < typ.NumField(); idx++ {
		switch typ.Field(idx).Name {
		case "MinZeta":
			val.Elem().Field(idx).Set(reflect.ValueOf(conf.zeta(25)))
		case "MaxZeta":
			val.Elem().Field(idx).Set(reflect.ValueOf(conf.zeta(2.0408163265306123)))
		default:
			start, err := strconv.ParseFloat(typ.Field(idx).Tag.Get("start"), 64)
			if err != nil {
				log.Panicf("Field %+v doesn't have a start tag!", typ.Field(idx))
			}
			val.Elem().Field(idx).Set(reflect.ValueOf(start))
		}
	}
}

func (x *XValues) toNormalizedFloat64Slice(conf *OptConfig) []float64 {
	val := reflect.ValueOf(*x)
	result := []float64{}
	for _, idx := range x.activeFieldIndices(conf) {
		result = append(result, normalize(val.Field(idx).Float(), x.scaleForField(idx)))
	}
	return result
}

func (x *XValues) setFromNormalizedFloat64Slice(conf *OptConfig, xSlice []float64) {
	x.init(conf)
	val := reflect.ValueOf(x)
	for _, idx := range x.activeFieldIndices(conf) {
		val.Elem().Field(idx).Set(reflect.ValueOf(denormalize(xSlice[0], x.scaleForField(idx))))
		xSlice = xSlice[1:]
	}
}

type Evaluation struct {
	Signal            signals.Float64Slice
	EvaluationID      string
	RunID             string
	ProbeSampler      signals.Noise
	EvaluatedLoudness signals.DB
}

type evaluations []Evaluation

func (e evaluations) Less(i, j int) bool {
	return bytes.Compare([]byte(e[i].EvaluationID), []byte(e[j].EvaluationID)) < 0
}

func (e evaluations) Len() int {
	return len(e)
}

func (e evaluations) Swap(i, j int) {
	e[i], e[j] = e[j], e[i]
}

type psnr struct {
	evaluation        Evaluation
	psnr              float64
	predictedLoudness signals.DB
}

type psnrs []psnr

func (p psnrs) Len() int {
	return len(p)
}

func (p psnrs) Less(i, j int) bool {
	iParts := strings.Split(p[i].evaluation.EvaluationID, "_")
	jParts := strings.Split(p[j].evaluation.EvaluationID, "_")
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

type remoteComputer struct {
	spec      string
	client    *rpc.Client
	available int64
}

type LossCalculator struct {
	Conf        *OptConfig
	evaluations evaluations

	outDir                     string
	lossCalculationOutputRatio int
	remoteComputers            []remoteComputer

	err              error
	lossCalculations int
}

func (l *LossCalculator) loadEvaluations(glob string) error {
	l.evaluations = nil

	wp := workerpool.New(map[string]int{"L": runtime.NumCPU(), "P": 0, "C": 0})

	evaluationChannel := make(chan Evaluation)
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
							eval := Evaluation{
								EvaluatedLoudness: signals.DB(equivalentLoudness.Results.ProbeDBSPLForEquivalentLoudness),
								EvaluationID:      equivalentLoudness.Evaluation.ID,
								RunID:             equivalentLoudness.Run.ID,
							}
							probeSampler, err := equivalentLoudness.Evaluation.Probe.Sampler()
							if err != nil {
								return err
							}
							probeNoise, ok := probeSampler.(*signals.Noise)
							if !ok {
								return fmt.Errorf("Probe sampler %+v isn't a Noise sampler?", probeSampler)
							}
							eval.ProbeSampler = *probeNoise
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
							noisySampler := signals.Superposition{combinedSampler, &signals.Noise{Color: signals.White, LowerLimit: 20, UpperLimit: 20000, Level: l.Conf.NoiseFloor - l.Conf.EvaluationFullScaleSineLevel}}
							eval.Signal, err = noisySampler.Sample(signals.TimeStretch{FromInclusive: 0, ToExclusive: 1}, rate, nil)
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
	if err := wp.WaitAll(); err != nil {
		return err
	}
	sort.Sort(l.evaluations)
	return nil
}

func (l *LossCalculator) loss(x []float64) float64 {
	return l.lossHelper(x, "", "")
}

type ComputePSNRReq struct {
	EvaluationIndex int
	XValues         XValues
}

type ComputePSNRResp struct {
	PSNR              float64
	PredictedLoudness signals.DB
}

type ChecksumResp struct {
	Evaluations []byte
	Config      []byte
}

func (l *LossCalculator) Checksum(req struct{}, result *ChecksumResp) error {
	h := sha1.New()
	if err := json.NewEncoder(h).Encode(l); err != nil {
		return err
	}
	result.Config = h.Sum(nil)
	h = sha1.New()
	if err := json.NewEncoder(h).Encode(l.evaluations); err != nil {
		return err
	}
	result.Evaluations = h.Sum(nil)
	return nil
}

func (l *LossCalculator) NumCPU(req struct{}, result *int) error {
	*result = runtime.NumCPU()
	return nil
}

func (l *LossCalculator) ComputePSNR(req ComputePSNRReq, resp *ComputePSNRResp) error {
	evaluation := l.evaluations[req.EvaluationIndex]
	carfacParams := carfac.CARFACParams{
		SampleRate: rate,

		VelocityScale: &req.XValues.VelocityScale,
		VOffset:       &req.XValues.VOffset,
		MinZeta:       &req.XValues.MinZeta,
		MaxZeta:       &req.XValues.MaxZeta,
		ZeroRatio:     &req.XValues.ZeroRatio,
		ERBPerStep:    &l.Conf.ERBPerStep,

		StageGain:       &req.XValues.StageGain,
		AGC1Scale0:      &req.XValues.AGC1Scale0,
		AGC1ScaleMul:    &req.XValues.AGC1ScaleMul,
		AGC2Scale0:      &req.XValues.AGC2Scale0,
		AGC2ScaleMul:    &req.XValues.AGC2ScaleMul,
		TimeConstant0:   &req.XValues.TimeConstant0,
		TimeConstantMul: &req.XValues.TimeConstantMul,
	}
	cf := carfac.New(carfacParams)
	// The runtime finalizer will run this automatically, but to speed up the cleanup.
	defer cf.Destroy()
	scaledSignal := evaluation.Signal.ToFloat32AddLevel(l.Conf.EvaluationFullScaleSineLevel - signals.DB(req.XValues.CarfacFullScaleSineLevel))
	cf.Run(scaledSignal[len(evaluation.Signal)-cf.NumSamples():])
	if l.Conf.OpenLoop {
		cf.RunOpen(scaledSignal[len(evaluation.Signal)-cf.NumSamples():])
	}
	var cfOut []float32
	var err error
	if !l.Conf.UsingNAP {
		cfOut, err = cf.BM()
	} else {
		cfOut, err = cf.NAP()
	}
	if err != nil {
		return err
	}
	resp.PSNR = -math.MaxFloat64
	for chanIdx := 0; chanIdx < cf.NumChannels(); chanIdx++ {
		channel := make([]float64, fftWindowSize)
		for sampleIdx := range channel {
			channel[sampleIdx] = float64(cfOut[(cf.NumSamples()-fftWindowSize+sampleIdx)*cf.NumChannels()+chanIdx])
		}
		channelPower := 0.0
		spec := spectrum.S{}
		if l.Conf.UseSNNR {
			channelPower = 10 * math.Log10(stat.Variance(channel, nil))
			spec = spectrum.ComputeSignalPower(channel, rate)
		} else {
			spec = spectrum.Compute(channel, rate)
		}
		for binIdx := int(math.Floor(float64(evaluation.ProbeSampler.LowerLimit / spec.BinWidth))); binIdx <= int(math.Ceil(float64(evaluation.ProbeSampler.UpperLimit/spec.BinWidth))); binIdx++ {
			snr := 0.0
			if l.Conf.UseSNNR {
				snr = spec.SignalPower[binIdx] - channelPower
			} else {
				snr = spec.SignalPower[binIdx] - spec.NoisePower[binIdx]
			}
			if snr > resp.PSNR {
				resp.PSNR = snr
			}
		}
	}
	resp.PredictedLoudness = signals.DB(req.XValues.LoudnessConstant + req.XValues.LoudnessScale*resp.PSNR)
	return nil
}

func (l *LossCalculator) lossHelper(x []float64, forceLogWorstTo string, forceLogAllTo string) float64 {
	l.lossCalculations++
	xv := XValues{}
	xv.setFromNormalizedFloat64Slice(l.Conf, x)
	fmt.Printf("Evaluation ** %v ** using %+v with %s\n", l.lossCalculations, l.Conf, xv.activeValues(l.Conf))

	bar := pb.StartNew(len(l.evaluations)).Prefix("Evaluating")
	psnrChan := make(chan psnr, len(l.evaluations))
	availableComputers := 0
	for _, rc := range l.remoteComputers {
		availableComputers += int(rc.available)
	}
	wp := workerpool.New(map[string]int{"L": availableComputers, "P": 0})
	wp.Queue("P", func() error {
		for evaluationIdxVar := range l.evaluations {
			evaluationIdx := evaluationIdxVar
			wp.Queue("L", func() error {
				var rcToUse *remoteComputer
				for _, rcIdx := range rand.Perm(len(l.remoteComputers)) {
					if availablePostTake := atomic.AddInt64(&l.remoteComputers[rcIdx].available, -1); availablePostTake >= 0 {
						rcToUse = &l.remoteComputers[rcIdx]
						defer atomic.AddInt64(&l.remoteComputers[rcIdx].available, 1)
					} else {
						atomic.AddInt64(&l.remoteComputers[rcIdx].available, 1)
					}
				}
				if rcToUse == nil {
					log.Fatal("Didn't find a remote computer to use, this is unheard of?")
				}

				resp := ComputePSNRResp{}
				if err := rcToUse.client.Call("LossCalculator.ComputePSNR", ComputePSNRReq{EvaluationIndex: evaluationIdx, XValues: xv}, &resp); err != nil {
					log.Fatalf("Unable to call LossCalculator.ComputePSNR using %+v: %v", rcToUse, err)
				}

				psnrChan <- psnr{
					evaluation:        l.evaluations[evaluationIdx],
					psnr:              resp.PSNR,
					predictedLoudness: resp.PredictedLoudness,
				}
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
		psnrsByRunID[evalPSNR.evaluation.RunID] = append(psnrsByRunID[evalPSNR.evaluation.RunID], evalPSNR)
		predictedLoudnessError := evalPSNR.predictedLoudness - evalPSNR.evaluation.EvaluatedLoudness
		if evalPSNR.evaluation.EvaluatedLoudness < 27 {
			errorDiscount := signals.DB(30-evalPSNR.evaluation.EvaluatedLoudness) / 3
			predictedLoudnessError /= errorDiscount
		}
		square := math.Pow(float64(predictedLoudnessError*predictedLoudnessError), l.Conf.PNorm)
		sumOfSquares += square
		lossByRunID[evalPSNR.evaluation.RunID] += square
	}
	worstRun := psnrs{}
	worstAvgLoss := 0.0
	for runID := range lossByRunID {
		lossByRunID[runID] = math.Pow(lossByRunID[runID]/float64(len(psnrsByRunID[runID])), 1.0/l.Conf.PNorm)
		if lossByRunID[runID] > worstAvgLoss {
			worstAvgLoss = lossByRunID[runID]
			worstRun = psnrsByRunID[runID]
		}
	}
	if len(l.remoteComputers) > 0 {
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
	}
	loss := math.Pow(sumOfSquares/float64(len(l.evaluations)), 1.0/l.Conf.PNorm)
	limitLoss := 0.0
	explanation := []string{"limit loss disabled"}
	if l.Conf.Limits {
		limitLoss, explanation = xv.limitLoss()
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

func (l *LossCalculator) makePythonPlot(filename string, title string, xlabel string, ylabel string, plots []plot) error {
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

func (l *LossCalculator) logPSNRs(p psnrs, name string) error {
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
		evaluatedYValues = append(evaluatedYValues, float64(p[idx].evaluation.EvaluatedLoudness))
		predictedYValues = append(predictedYValues, float64(p[idx].predictedLoudness))
	}
	if err := l.makePythonPlot(logFile, fmt.Sprintf("Optimization run %v/Evaluation run %v", l.lossCalculations, p[0].evaluation.RunID), "Evaluation index", "Loudness", []plot{
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
		conf := &OptConfig{UsingNAP: usingNAP}
		testXValues := XValues{}
		xSlice := testXValues.toNormalizedFloat64Slice(conf)
		for idx := range xSlice {
			xSlice[idx] = float64(idx)
		}
		preRoundtripValues := &XValues{}
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

func (l *LossCalculator) optimize() error {
	problem := optimize.Problem{
		Func: l.loss,
		Status: func() (optimize.Status, error) {
			return optimize.NotTerminated, l.err
		},
	}
	initX := XValues{}
	if *startX == "" {
		initX.init(l.Conf)
	} else {
		if err := json.Unmarshal([]byte(*startX), initX); err != nil {
			return err
		}
	}
	res, err := optimize.Minimize(problem, initX.toNormalizedFloat64Slice(l.Conf), nil, nil)
	if err != nil {
		return err
	}

	resultValues := XValues{}
	resultValues.setFromNormalizedFloat64Slice(l.Conf, res.Location.X)
	finalFile, err := os.Create(filepath.Join(*outputDir, "final_results.json"))
	if err != nil {
		return err
	}
	defer finalFile.Close()
	if err := json.NewEncoder(finalFile).Encode(map[string]interface{}{
		"X":    resultValues,
		"Conf": l.Conf,
		"Loss": l.lossHelper(resultValues.toNormalizedFloat64Slice(l.Conf), "worst_evaluation_run_final_results", "all_evaluation_runs_final_results"),
	}); err != nil {
		return err
	}

	b, err := json.Marshal(resultValues)
	if err != nil {
		return err
	}
	fmt.Printf("Final results: %+v\n%v\n%s\n", res.Stats, res.Status, b)
	return nil
}

func (l *LossCalculator) explore(field string, points int) {
	initX := XValues{}
	initX.init(l.Conf)
	initXTyp := reflect.TypeOf(initX)
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
		loss := l.loss(initX.toNormalizedFloat64Slice(l.Conf))
		losses = append(losses, loss)
	}
	logFile := filepath.Join(l.outDir, fmt.Sprintf("exploration_of_%s_over_%v_points.py", field, points))
	if err := l.makePythonPlot(logFile, fmt.Sprintf("Exploration of loss over %s", field), field, "Loss", []plot{{x: vals, y: losses, label: "Loss"}}); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Plotted exploration of %v over %v points to %v\n", field, points, logFile)
}

func (l *LossCalculator) serveOrOptimize() error {
	if err := l.loadEvaluations(*evaluationJSONGlob); err != nil {
		return err
	}
	if len(l.remoteComputers) == 0 {
		rpc.Register(l)
		rpc.HandleHTTP()
		log.Printf("Listening on :8080 for connections...")
		http.ListenAndServe("0.0.0.0:8080", nil)
	} else {
		localSum := ChecksumResp{}
		if err := l.Checksum(struct{}{}, &localSum); err != nil {
			log.Fatal(err)
		}
		wp := workerpool.New(map[string]int{"P": 0})
		for _, rcVar := range l.remoteComputers {
			rc := rcVar
			wp.Queue("P", func() error {
				remoteSum := ChecksumResp{}
				if err := rc.client.Call("LossCalculator.Checksum", struct{}{}, &remoteSum); err != nil {
					return err
				}
				if bytes.Compare(localSum.Config, remoteSum.Config) != 0 {
					return fmt.Errorf("Remote computer %q has different config checksum than local client!", rc.spec)
				}
				if bytes.Compare(localSum.Evaluations, remoteSum.Evaluations) != 0 {
					return fmt.Errorf("Remote computer %q has different input checksum than local client!", rc.spec)
				}
				return nil
			})
		}
		wp.Close("P")
		if err := wp.WaitAll(); err != nil {
			return err
		}
		if err := os.MkdirAll(*outputDir, 0755); err != nil {
			return err
		}

		return l.optimize()
	}
	return nil
}

func main() {
	test()

	flag.Parse()
	if *evaluationJSONGlob == "" {
		flag.Usage()
		os.Exit(1)
	}
	lc := &LossCalculator{
		outDir: *outputDir,
		Conf: &OptConfig{
			PNorm:                        *pNorm,
			OpenLoop:                     !*skipOpenLoop,
			UsingNAP:                     !*usingBM,
			DisabledFields:               map[string]bool{},
			Limits:                       !*noLimits,
			UseSNNR:                      *useSNNR,
			ERBPerStep:                   *erbPerStep,
			EvaluationFullScaleSineLevel: signals.DB(*evaluationFullScaleSineLevel),
			NoiseFloor:                   signals.DB(*noiseFloor),
		},
		lossCalculationOutputRatio: *lossCalculationOutputRatio,
	}
	for _, remoteSpec := range strings.Split(*remoteComputers, ",") {
		if remoteSpec != "" {
			client, err := rpc.DialHTTP("tcp", remoteSpec)
			if err != nil {
				log.Fatal(err)
			}
			rc := remoteComputer{
				spec:   remoteSpec,
				client: client,
			}
			if err := rc.client.Call("LossCalculator.NumCPU", struct{}{}, &rc.available); err != nil {
				log.Fatal(err)
			}
			lc.remoteComputers = append(lc.remoteComputers, rc)
		}
	}
	for _, disabledField := range strings.Split(*disabledFields, ",") {
		if strings.TrimSpace(disabledField) != "" {
			lc.Conf.DisabledFields[disabledField] = true
		}
	}

	if *exploreField != "" {
		lc.explore(*exploreField, *exploreFieldPoints)
	} else {
		if err := lc.serveOrOptimize(); err != nil {
			log.Fatal(err)
		}
	}
}
