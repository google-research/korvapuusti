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
	"bytes"
	"crypto/sha1"
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
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
	"runtime/pprof"
	"sort"
	"strconv"
	"strings"

	"github.com/cheggaaa/pb"
	"github.com/google-research/korvapuusti/experiments/partial_loudness/analysis"
	"github.com/google-research/korvapuusti/tools/carfac"
	"github.com/google-research/korvapuusti/tools/loudness"
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

func init() {
	gob.Register(map[string]interface{}{})
	gob.Register([]interface{}{})
}

var (
	sqrt2 = math.Sqrt(2.0)

	// Optimize-time outputs.

	outputDir                  = flag.String("output_dir", filepath.Join(os.Getenv("HOME"), "optimize_carfac_for_psnr"), "Directory to put output files in.")
	lossCalculationOutputRatio = flag.Int("loss_calculation_output_ratio", 100, "How seldom to output data about the best/worst results of a calculation run.")
	remoteComputers            = flag.String("remote_computers", "", "Comma separated list of host:port pairs defining remote computers to use. Not providing this will instead run a remote computer unless run_local is provided.")
	runLocal                   = flag.Bool("run_local", false, "Run locally instead of using the distributed mode.")
	cpuprofile                 = flag.String("cpuprofile", "", "Write cpu profile to file.")

	// Definition of evaluations.

	evaluationJSONGlob           = flag.String("evaluation_json_glob", "", "Glob to the files containing evaluations.")
	evaluationFullScaleSineLevel = flag.Float64("evaluation_full_scale_sine_level", 100.0, "The calibration for a full scale sine during evaluation.")
	noiseFloor                   = flag.Float64("noise_floor", 35, "Noise floor when evaluations were made.")
	mergeEvaluations             = flag.Bool("merge_evaluations", true, "Merge evaluations of identical probe/masker combinations.")

	// Optimization settings.

	startX         = flag.String("start_x", "", "Starting values in JSON format.")
	pNorm          = flag.Float64("p_norm", 4, "Power of the norm when calculating the loss.")
	skipOpenLoop   = flag.Bool("skip_open_loop", false, "Whether to skip the second (open, less non-linear) run of each signal sample.")
	usingBM        = flag.Bool("using_bm", true, "Whether to use the basilar membrane output of CARFAC (as opposed to the neural activation pattern output).")
	disabledFields = flag.String("disabled_fields", "", "Comma separated fields to avoid optimizing (leave at start value).")
	noLimits       = flag.Bool("no_limits", false, "Disable the limit loss.")
	useSNNR        = flag.Bool("use_snnr", true, "Use SNNR instead of SNR to estimate partial loudness.")
	erbPerStep     = flag.Float64("erb_per_step", 0.01, "erb_per_step while running CARFAC.")
	useGaussianSum = flag.Bool("use_gaussian_sum", false, "Whether to use a gaussian sum of SNRs centered around the output frequency instead of the SNR of the output frequency when predicting loudness.")

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
	OpenLoop                     bool
	UsingNAP                     bool
	DisabledFields               map[string]bool
	PNorm                        float64
	Limits                       bool
	UseSNNR                      bool
	ERBPerStep                   float64
	EvaluationFullScaleSineLevel signals.DB
	NoiseFloor                   signals.DB
	MergeEvaluations             bool
	UseGaussianSum               bool
}

func (o *optConfig) zeta(zetaConst float64) float64 {
	// Based on the assumtion that max small-signal gain at the passband peak
	// will be on the order of  (0.5/min_zeta)^(1/erb_per_step), and we need
	// the start value of that in the same region or the loss function becomes
	// too uneven to optimize.
	return math.Pow(zetaConst, -o.ERBPerStep) * math.Pow(math.Pow(0.5, -1.0/o.ERBPerStep), -o.ERBPerStep)
}

type XValues struct {
	CarfacFullScaleSineLevel float64 `start:"100.0" scale:"70.0,130.0" limits:"-,-"`

	VelocityScale float64 `start:"0.1" scale:"0.02,0.5" limits:"0.0,-"`
	VOffset       float64 `start:"0.04" scale:"0.0,0.5" limits:"0.0,-"`
	MinZeta       float64 `scale:"0.01,0.5" limits:"0.0,-"`
	MaxZeta       float64 `scale:"0.1,5.0" limits:"-,-"`
	ZeroRatio     float64 `start:"1.4142135623730951" scale:"1.2,3.0" limits:"0.0,-"`

	StageGain       float64 `start:"2.0" scale:"1.2,8.0" limits:"1.2,-"`
	AGC1Scale0      float64 `start:"1.0" scale:"0.5,3.0" limits:"0.0,-"`
	AGC1ScaleMul    float64 `start:"1.4142135623730951" scale:"1.2,3.0" limits:"0.0,-"`
	AGC2Scale0      float64 `start:"1.65" scale:"0.2,2.0" limits:"0.0,-"`
	AGC2ScaleMul    float64 `start:"1.4142135623730951" scale:"1.2,3.0" limits:"0.0,-"`
	TimeConstant0   float64 `start:"0.002" scale:"0.001,0.008" limits:"0.0,-"`
	TimeConstantMul float64 `start:"4" scale:"2.0,8.0" limits:"0.0,-"`

	GaussianStdDev   float64 `start:"1.0" scale:"0.2,20" limits:"-,-" gaussian:"true"`
	LoudnessConstant float64 `start:"40.0" scale:"0.0,80.0" limits:"-,-"`
	LoudnessScale    float64 `start:"2.0" scale:"0.1,10.0" limits:"-,-"`
}

func (x *XValues) activeValues(conf *optConfig) string {
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

func (x *XValues) activeFieldIndices(conf *optConfig) []int {
	typ := reflect.TypeOf(*x)
	result := []int{}
	for idx := 0; idx < typ.NumField(); idx++ {
		if !conf.DisabledFields[typ.Field(idx).Name] && (conf.UseGaussianSum || typ.Field(idx).Tag.Get("gaussian") != "true") {
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

func (x *XValues) init(conf *optConfig) {
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

func (x *XValues) toNormalizedFloat64Slice(conf *optConfig) []float64 {
	val := reflect.ValueOf(*x)
	result := []float64{}
	for _, idx := range x.activeFieldIndices(conf) {
		result = append(result, normalize(val.Field(idx).Float(), x.scaleForField(idx)))
	}
	return result
}

func (x *XValues) setFromNormalizedFloat64Slice(conf *optConfig, xSlice []float64) {
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
	spec   string
	client *rpc.Client
	numCPU int
}

type remoteComputerSlice []remoteComputer

func (r remoteComputerSlice) shuffledPool() chan *remoteComputer {
	slice := []*remoteComputer{}
	for i := range r {
		for j := 0; j < r[i].numCPU; j++ {
			slice = append(slice, &r[i])
		}
	}
	rand.Shuffle(len(slice), func(i, j int) { slice[i], slice[j] = slice[j], slice[i] })
	result := make(chan *remoteComputer, len(slice))
	for _, rc := range slice {
		result <- rc
	}
	return result
}

type LossCalculator struct {
	conf                       *optConfig
	evaluationJSONGlob         string
	equivalentLoudnesses       analysis.EquivalentLoudnesses
	evaluations                evaluations
	outDir                     string
	runLocal                   bool
	lossCalculationOutputRatio int
	remoteComputers            remoteComputerSlice

	err              error
	lossCalculations int
}

func (l *LossCalculator) loadEvaluations() error {
	l.equivalentLoudnesses = nil

	evaluationFileNames, err := filepath.Glob(l.evaluationJSONGlob)
	if err != nil {
		return err
	}

	for _, evaluationFileName := range evaluationFileNames {
		equivs := analysis.EquivalentLoudnesses{}
		evaluationFile, err := os.Open(evaluationFileName)
		if err != nil {
			return err
		}
		defer evaluationFile.Close()
		if err := equivs.LoadAppend(evaluationFile); err != nil {
			return err
		}
		for _, equiv := range equivs {
			if equiv.EntryType == "EquivalentLoudnessMeasurement" {
				l.equivalentLoudnesses = append(l.equivalentLoudnesses, equiv)
			}
		}
	}
	if l.conf.MergeEvaluations {
		merged, err := l.equivalentLoudnesses.Merge()
		if err != nil {
			return err
		}
		fmt.Printf("Merged %v evaluations into %v\n", len(l.equivalentLoudnesses), len(merged))
		l.equivalentLoudnesses = merged
	}
	return nil
}

func (l *LossCalculator) SynthesizeEvaluations(req struct{}, checksum *[]byte) error {
	l.evaluations = nil

	evaluationChannel := make(chan Evaluation)
	evaluationCollectionDone := make(chan struct{})
	go func() {
		for evaluation := range evaluationChannel {
			l.evaluations = append(l.evaluations, evaluation)
		}
		close(evaluationCollectionDone)
	}()

	fmt.Printf("Synthesizing with %+v\n", l.conf)
	cpuPool := workerpool.New(runtime.NumCPU())
	bar := pb.StartNew(len(l.equivalentLoudnesses)).Prefix("Synthesizing")
	overlapSkip := 0
	for _, equivalentLoudnessVar := range l.equivalentLoudnesses {
		equivalentLoudness := equivalentLoudnessVar
		cpuPool.Go(func() error {
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
					overlapSkip++
					return nil
				}
				if noise.LowerLimit >= probeNoise.LowerLimit && noise.LowerLimit <= probeNoise.UpperLimit {
					overlapSkip++
					return nil
				}
				if noise.UpperLimit >= probeNoise.LowerLimit && noise.UpperLimit <= probeNoise.UpperLimit {
					overlapSkip++
					return nil
				}
			}
			noisySampler := signals.Superposition{combinedSampler, &signals.Noise{Color: signals.White, LowerLimit: 20, UpperLimit: 20000, Level: l.conf.NoiseFloor - l.conf.EvaluationFullScaleSineLevel}}
			eval.Signal, err = noisySampler.Sample(signals.TimeStretch{FromInclusive: 0, ToExclusive: 1}, rate, nil)
			if err != nil {
				return err
			}
			evaluationChannel <- eval
			bar.Increment()
			return nil
		})
	}
	if err := cpuPool.Wait(); err != nil {
		return err
	}
	close(evaluationChannel)
	<-evaluationCollectionDone
	bar.Finish()

	fmt.Printf("Skipped %v signals due to overlap with masker. Checksumming %v signals.\n", overlapSkip, len(l.evaluations))
	sort.Sort(l.evaluations)
	h := sha1.New()
	if err := json.NewEncoder(h).Encode(l.evaluations); err != nil {
		return err
	}
	*checksum = h.Sum(nil)

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

type ConfigureReq struct {
	Conf                 optConfig
	EquivalentLoudnesses []analysis.EquivalentLoudness
}

func (l *LossCalculator) Configure(req ConfigureReq, res *struct{}) error {
	l.conf = &req.Conf
	l.equivalentLoudnesses = req.EquivalentLoudnesses
	return nil
}

type ChecksumResp struct {
	Evaluations []byte
}

func (l *LossCalculator) Checksum(req struct{}, result *ChecksumResp) error {
	h := sha1.New()
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
		ERBPerStep:    &l.conf.ERBPerStep,

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
	scaledSignal := evaluation.Signal.ToFloat32AddLevel(l.conf.EvaluationFullScaleSineLevel - signals.DB(req.XValues.CarfacFullScaleSineLevel))
	cf.Run(scaledSignal[len(evaluation.Signal)-cf.NumSamples():])
	if l.conf.OpenLoop {
		cf.RunOpen(scaledSignal[len(evaluation.Signal)-cf.NumSamples():])
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
	resp.PSNR = -math.MaxFloat64
	resp.PredictedLoudness = -math.MaxFloat64
	for chanIdx := 0; chanIdx < cf.NumChannels(); chanIdx++ {
		channel := make([]float64, fftWindowSize)
		for sampleIdx := range channel {
			channel[sampleIdx] = float64(cfOut[(cf.NumSamples()-fftWindowSize+sampleIdx)*cf.NumChannels()+chanIdx])
		}
		channelPower := 0.0
		spec := spectrum.S{}
		if l.conf.UseSNNR {
			channelPower = 10 * math.Log10(stat.Variance(channel, nil))
			spec = spectrum.ComputeSignalPower(channel, rate)
		} else {
			spec = spectrum.Compute(channel, rate)
		}
		for binIdx := int(math.Floor(float64(evaluation.ProbeSampler.LowerLimit / spec.BinWidth))); binIdx <= int(math.Ceil(float64(evaluation.ProbeSampler.UpperLimit/spec.BinWidth))); binIdx++ {
			binSnr := 0.0
			if l.conf.UseGaussianSum {
				for componentBinIdx := range spec.SignalPower {
					componentSNR := 0.0
					if l.conf.UseSNNR {
						componentSNR = spec.SignalPower[componentBinIdx] - channelPower
					} else {
						componentSNR = spec.SignalPower[componentBinIdx] - spec.NoisePower[componentBinIdx]
					}
					binSnr += componentSNR * math.Exp(-math.Pow(float64(componentBinIdx-binIdx), 2)/(2*math.Pow(req.XValues.GaussianStdDev, 2)))
				}
			} else {
				if l.conf.UseSNNR {
					binSnr = spec.SignalPower[binIdx] - channelPower
				} else {
					binSnr = spec.SignalPower[binIdx] - spec.NoisePower[binIdx]
				}
			}
			if binSnr > resp.PSNR {
				resp.PSNR = binSnr
			}
			binCFLoudness := req.XValues.LoudnessConstant + req.XValues.LoudnessScale*binSnr
			binSPLLoudness := loudness.Phons2SPL(binCFLoudness, float64(binIdx)*float64(spec.BinWidth))
			if binSPLLoudness > float64(resp.PredictedLoudness) {
				resp.PredictedLoudness = signals.DB(binSPLLoudness)
			}
		}
	}
	return nil
}

func (l *LossCalculator) lossHelper(x []float64, forceLogWorstTo string, forceLogAllTo string) float64 {
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer func() {
			pprof.StopCPUProfile()
			f.Close()
			fmt.Printf("Stored cpu profile in %v\n", *cpuprofile)
			os.Exit(0)
		}()
	}
	xv := XValues{}
	xv.setFromNormalizedFloat64Slice(l.conf, x)
	fmt.Printf("Evaluation ** %v ** using %+v with %s\n", l.lossCalculations, l.conf, xv.activeValues(l.conf))

	bar := pb.StartNew(len(l.evaluations)).Prefix("Evaluating")
	psnrChan := make(chan psnr, len(l.evaluations))
	shuffledComputers := l.remoteComputers.shuffledPool()
	wp := workerpool.New(0)
	evaluationsDone := make(chan struct{})
	wp.Go(func() error {
		for evaluationIdxVar := range l.evaluations {
			evaluationIdx := evaluationIdxVar
			wp.Go(func() error {
				resp := ComputePSNRResp{}
				if l.runLocal {
					if err := l.ComputePSNR(ComputePSNRReq{EvaluationIndex: evaluationIdx, XValues: xv}, &resp); err != nil {
						log.Fatal("Unable to call ComputePSNR locally: %v", err)
					}
				} else {
					rcToUse := <-shuffledComputers
					err := rcToUse.client.Call("LossCalculator.ComputePSNR", ComputePSNRReq{EvaluationIndex: evaluationIdx, XValues: xv}, &resp)
					shuffledComputers <- rcToUse
					if err != nil {
						log.Fatalf("Unable to call LossCalculator.ComputePSNR using %+v: %v", rcToUse, err)
					}
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
		close(evaluationsDone)
		return nil
	})
	<-evaluationsDone
	if err := wp.Wait(); err != nil {
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
		predictedLoudnessError := float64(evalPSNR.predictedLoudness - evalPSNR.evaluation.EvaluatedLoudness)
		if evalPSNR.evaluation.EvaluatedLoudness < l.conf.NoiseFloor-8 {
			errorDiscount := float64(l.conf.NoiseFloor-5-evalPSNR.evaluation.EvaluatedLoudness) / 3.0
			predictedLoudnessError /= errorDiscount
		}
		square := math.Pow(predictedLoudnessError, l.conf.PNorm)
		sumOfSquares += square
		lossByRunID[evalPSNR.evaluation.RunID] += square
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
	if l.runLocal || len(l.remoteComputers) > 0 {
		if l.lossCalculations%l.lossCalculationOutputRatio == 0 {
			if err := l.logPSNRs(worstRun, "worst_evaluation_run"); err != nil {
				l.err = err
				return 0.0
			}
			all := psnrs{}
			for _, runs := range psnrsByRunID {
				all = append(all, runs...)
			}
			if err := l.logPSNRs(all, "all_evaluation_runs"); err != nil {
				l.err = err
				return 0.0
			}
		}
		if forceLogWorstTo != "" {
			if err := l.logPSNRs(worstRun, forceLogWorstTo); err != nil {
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
	loss := math.Pow(sumOfSquares/float64(len(l.evaluations)), 1.0/l.conf.PNorm)
	limitLoss := 0.0
	explanation := []string{"limit loss disabled"}
	if l.conf.Limits {
		limitLoss, explanation = xv.limitLoss()
	}
	totalLoss := loss + limitLoss
	fmt.Printf("Got loss %v (limit loss %v: %v)\n", totalLoss, limitLoss, strings.Join(explanation, ", "))
	l.lossCalculations++
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
	for _, useGaussianSum := range []bool{true, false} {
		conf := &optConfig{UseGaussianSum: useGaussianSum}
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

type remoteChecksum struct {
	spec     string
	checksum []byte
}

func (l *LossCalculator) verifyRemotes() error {
	localSum := []byte{}
	wp := workerpool.New(0)
	wp.Go(func() error {
		return l.SynthesizeEvaluations(struct{}{}, &localSum)
	})
	otherSums := make(chan remoteChecksum, len(l.remoteComputers))
	for _, rcVar := range l.remoteComputers {
		rc := rcVar
		wp.Go(func() error {
			if err := rc.client.Call("LossCalculator.Configure", ConfigureReq{
				Conf:                 *l.conf,
				EquivalentLoudnesses: l.equivalentLoudnesses,
			}, &struct{}{}); err != nil {
				log.Printf("failed configure: %v", err)
				return err
			}
			remoteSum := []byte{}
			if err := rc.client.Call("LossCalculator.SynthesizeEvaluations", struct{}{}, &remoteSum); err != nil {
				return err
			}
			otherSums <- remoteChecksum{spec: rc.spec, checksum: remoteSum}
			return nil
		})
	}
	if err := wp.Wait(); err != nil {
		return err
	}
	close(otherSums)
	for otherSum := range otherSums {
		if bytes.Compare(localSum, otherSum.checksum) != 0 {
			return fmt.Errorf("remote computer %v has different evaluations checksum from controller", otherSum.spec)
		}
	}
	return nil
}

func (l *LossCalculator) optimize() error {
	if l.runLocal {
		localSum := []byte{}
		if err := l.SynthesizeEvaluations(struct{}{}, &localSum); err != nil {
			return err
		}

	} else {
		if err := l.verifyRemotes(); err != nil {
			return err
		}
	}
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		return err
	}

	problem := optimize.Problem{
		Func: l.loss,
		Status: func() (optimize.Status, error) {
			return optimize.NotTerminated, l.err
		},
	}
	initX := XValues{}
	if *startX == "" {
		initX.init(l.conf)
	} else {
		if err := json.Unmarshal([]byte(*startX), &initX); err != nil {
			return err
		}
	}
	res, err := optimize.Minimize(problem, initX.toNormalizedFloat64Slice(l.conf), nil, nil)
	if err != nil {
		return err
	}

	resultValues := XValues{}
	resultValues.setFromNormalizedFloat64Slice(l.conf, res.Location.X)
	finalFile, err := os.Create(filepath.Join(*outputDir, "final_results.json"))
	if err != nil {
		return err
	}
	defer finalFile.Close()
	if err := json.NewEncoder(finalFile).Encode(map[string]interface{}{
		"X":                  resultValues,
		"Conf":               l.conf,
		"Loss":               l.lossHelper(resultValues.toNormalizedFloat64Slice(l.conf), "worst_evaluation_run_final_results", "all_evaluation_runs_final_results"),
		"EvaluationJSONGlob": l.evaluationJSONGlob,
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
	initX.init(l.conf)
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
	lc := &LossCalculator{
		conf: &optConfig{
			PNorm:                        *pNorm,
			OpenLoop:                     !*skipOpenLoop,
			UsingNAP:                     !*usingBM,
			DisabledFields:               map[string]bool{},
			Limits:                       !*noLimits,
			UseSNNR:                      *useSNNR,
			ERBPerStep:                   *erbPerStep,
			EvaluationFullScaleSineLevel: signals.DB(*evaluationFullScaleSineLevel),
			NoiseFloor:                   signals.DB(*noiseFloor),
			MergeEvaluations:             *mergeEvaluations,
			UseGaussianSum:               *useGaussianSum,
		},
		outDir:                     *outputDir,
		evaluationJSONGlob:         *evaluationJSONGlob,
		runLocal:                   *runLocal,
		lossCalculationOutputRatio: *lossCalculationOutputRatio,
	}
	for _, disabledField := range strings.Split(*disabledFields, ",") {
		if strings.TrimSpace(disabledField) != "" {
			lc.conf.DisabledFields[disabledField] = true
		}
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
			if err := rc.client.Call("LossCalculator.NumCPU", struct{}{}, &rc.numCPU); err != nil {
				log.Fatal(err)
			}
			lc.remoteComputers = append(lc.remoteComputers, rc)
		}
	}

	if *exploreField == "" && !lc.runLocal && len(lc.remoteComputers) == 0 {
		rpc.Register(lc)
		rpc.HandleHTTP()
		log.Printf("Listening on :8080 for connections...")
		http.ListenAndServe("0.0.0.0:8080", nil)
	}

	if *evaluationJSONGlob == "" {
		flag.Usage()
		os.Exit(1)
	}
	if err := lc.loadEvaluations(); err != nil {
		log.Fatal(err)
	}
	if *exploreField != "" {
		lc.explore(*exploreField, *exploreFieldPoints)
	} else {
		if err := lc.optimize(); err != nil {
			log.Fatal(err)
		}
	}
}
