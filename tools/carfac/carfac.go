/* carfac contains a wrapper around the CARFAC hearing model.
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
//go:generate sh -c "c++ -O2 -march=haswell -c `pkg-config --libs --cflags eigen3` carfac/cpp/agc.h carfac/cpp/binaural_sai.cc carfac/cpp/car.h carfac/cpp/carfac.cc carfac/cpp/carfac_util.h carfac/cpp/common.h carfac/cpp/ear.cc carfac/cpp/ihc.h carfac/cpp/sai.cc glue/glue.cc && c++ -g -pg `pkg-config --libs --cflags eigen3` *.o -I. benchmark/benchmark.cc -o benchmark/benchmark && ar rcs libcarfac.a *.o && cd carfac && git clean -f && cd .."
package carfac

// #cgo CFLAGS: -I${SRCDIR}
// #cgo LDFLAGS: -L${SRCDIR} -lcarfac -lstdc++ -lm
// #include "carfac.h"
import "C"

import (
	"fmt"
	"reflect"
	"runtime"
	"unsafe"
)

func newFloatAry(length int) (C.float_ary, []float32) {
	floats := make([]float32, length)
	return C.float_ary{
		len:  C.int(length),
		data: (*C.float)(&floats[0]),
	}, floats
}

func floatAryToFloats(ary interface{}) []float32 {
	val := reflect.ValueOf(ary)
	var floats []float32
	header := (*reflect.SliceHeader)((unsafe.Pointer(&floats)))
	header.Cap = int(val.FieldByName("len").Int())
	header.Len = int(val.FieldByName("len").Int())
	header.Data = val.FieldByName("data").Pointer()
	return floats
}

func floatsToFloatAry(floats []float32) C.float_ary {
	return C.float_ary{
		len:  C.int(len(floats)),
		data: (*C.float)(&floats[0]),
	}
}

type CF interface {
	Run(buffer []float32)
	RunOpen(buffer []float32)
	Reset()
	NAP() ([]float32, error)
	BM() ([]float32, error)
	NumChannels() int
	NumSamples() int
	SampleRate() int
	Poles() []float32
	Destroy()
}

type carfac struct {
	numChannels int
	numSamples  int
	sampleRate  int
	poles       []float32
	openLoop    bool
	cf          *C.carfac
	destroyed   bool
}

func (c *carfac) NumChannels() int {
	return c.numChannels
}

func (c *carfac) NumSamples() int {
	return c.numSamples
}

func (c *carfac) SampleRate() int {
	return c.sampleRate
}

func (c *carfac) Poles() []float32 {
	return c.poles
}

type CARFACParams struct {
	SampleRate int

	VelocityScale           *float64
	VOffset                 *float64
	MinZeta                 *float64
	MaxZeta                 *float64
	ZeroRatio               *float64
	HighFDampingCompression *float64
	ERBPerStep              *float64
	ERBBreakFreq            *float64
	ERBQ                    *float64

	TauLPF     *float64
	Tau1Out    *float64
	Tau1In     *float64
	ACCornerHz *float64

	StageGain       *float64
	AGC1Scale0      *float64
	AGC1ScaleMul    *float64
	AGC2Scale0      *float64
	AGC2ScaleMul    *float64
	TimeConstant0   *float64
	TimeConstantMul *float64
	AGCMixCoeff     *float64
}

func (c *CARFACParams) Default(sampleRate int) *CARFACParams {
	c.SampleRate = sampleRate
	var cVelocityScale, cVOffset, cMinZeta, cMaxZeta, cZeroRatio, cHighFDampingCompression, cERBPerStep, cERBBreakFreq, cERBQ, cTauLPF, cTau1Out, cTau1In, cACCornerHz, cStageGain, cAGC1Scale0, cAGC1ScaleMul, cAGC2Scale0, cAGC2ScaleMul, cTimeConstant0, cTimeConstantMul, cAGCMixCoeff C.float
	C.set_default_params(
		&cVelocityScale,
		&cVOffset,
		&cMinZeta,
		&cMaxZeta,
		&cZeroRatio,
		&cHighFDampingCompression,
		&cERBPerStep,
		&cERBBreakFreq,
		&cERBQ,

		&cTauLPF,
		&cTau1Out,
		&cTau1In,
		&cACCornerHz,

		&cStageGain,
		&cAGC1Scale0,
		&cAGC1ScaleMul,
		&cAGC2Scale0,
		&cAGC2ScaleMul,
		&cTimeConstant0,
		&cTimeConstantMul,
		&cAGCMixCoeff,
	)
	velocityScale := float64(cVelocityScale)
	c.VelocityScale = &velocityScale
	vOffset := float64(cVOffset)
	c.VOffset = &vOffset
	minZeta := float64(cMinZeta)
	c.MinZeta = &minZeta
	maxZeta := float64(cMaxZeta)
	c.MaxZeta = &maxZeta
	zeroRatio := float64(cZeroRatio)
	c.ZeroRatio = &zeroRatio
	highFDampingCompression := float64(cHighFDampingCompression)
	c.HighFDampingCompression = &highFDampingCompression
	erbPerStep := float64(cERBPerStep)
	c.ERBPerStep = &erbPerStep
	erbBreakFreq := float64(cERBBreakFreq)
	c.ERBBreakFreq = &erbBreakFreq
	erbQ := float64(cERBQ)
	c.ERBQ = &erbQ
	tauLPF := float64(cTauLPF)
	c.TauLPF = &tauLPF
	tau1Out := float64(cTau1Out)
	c.Tau1Out = &tau1Out
	tau1In := float64(cTau1In)
	c.Tau1In = &tau1In
	acCornerHz := float64(cACCornerHz)
	c.ACCornerHz = &acCornerHz
	stageGain := float64(cStageGain)
	c.StageGain = &stageGain
	acg1Scale0 := float64(cAGC1Scale0)
	c.AGC1Scale0 = &acg1Scale0
	agc1ScaleMul := float64(cAGC1ScaleMul)
	c.AGC1ScaleMul = &agc1ScaleMul
	agc2Scale0 := float64(cAGC2Scale0)
	c.AGC2Scale0 = &agc2Scale0
	agc2ScaleMul := float64(cAGC2ScaleMul)
	c.AGC2ScaleMul = &agc2ScaleMul
	timeConstant0 := float64(cTimeConstant0)
	c.TimeConstant0 = &timeConstant0
	timeConstantMul := float64(cTimeConstantMul)
	c.TimeConstantMul = &timeConstantMul
	agcMixCoeff := float64(cAGCMixCoeff)
	c.AGCMixCoeff = &agcMixCoeff

	return c
}

func (c CARFACParams) cFloat(f *float64) *C.float {
	if f == nil {
		return nil
	}
	cFloat := C.float(*f)
	return &cFloat
}

func New(cfp CARFACParams) CF {
	cf := C.create_carfac(
		C.int(cfp.SampleRate),

		cfp.cFloat(cfp.VelocityScale),
		cfp.cFloat(cfp.VOffset),
		cfp.cFloat(cfp.MinZeta),
		cfp.cFloat(cfp.MaxZeta),
		cfp.cFloat(cfp.ZeroRatio),
		cfp.cFloat(cfp.HighFDampingCompression),
		cfp.cFloat(cfp.ERBPerStep),
		cfp.cFloat(cfp.ERBBreakFreq),
		cfp.cFloat(cfp.ERBQ),

		cfp.cFloat(cfp.TauLPF),
		cfp.cFloat(cfp.Tau1Out),
		cfp.cFloat(cfp.Tau1In),
		cfp.cFloat(cfp.ACCornerHz),

		cfp.cFloat(cfp.StageGain),
		cfp.cFloat(cfp.AGC1Scale0),
		cfp.cFloat(cfp.AGC1ScaleMul),
		cfp.cFloat(cfp.AGC2Scale0),
		cfp.cFloat(cfp.AGC2ScaleMul),
		cfp.cFloat(cfp.TimeConstant0),
		cfp.cFloat(cfp.TimeConstantMul),
		cfp.cFloat(cfp.AGCMixCoeff),
	)
	result := &carfac{
		numChannels: int(cf.num_channels),
		numSamples:  int(cf.num_samples),
		sampleRate:  cfp.SampleRate,
		poles:       floatAryToFloats(cf.poles),
		cf:          &cf,
		destroyed:   false,
	}
	runtime.SetFinalizer(result, func(i interface{}) {
		i.(*carfac).Destroy()
	})
	return result
}

func (c *carfac) Destroy() {
	if !c.destroyed {
		c.destroyed = true
		C.delete_carfac(c.cf)
	}
}

func (c *carfac) Reset() {
	C.carfac_reset(c.cf)
}

func (c *carfac) RunOpen(buffer []float32) {
	C.carfac_run(c.cf, floatsToFloatAry(buffer), 1)
}

func (c *carfac) Run(buffer []float32) {
	C.carfac_run(c.cf, floatsToFloatAry(buffer), 0)
}

func (c *carfac) NAP() (result []float32, err error) {
	resultAry, resultFloats := newFloatAry(c.numChannels * c.numSamples)
	if errnum := C.carfac_nap(c.cf, resultAry); errnum != 0 {
		return nil, fmt.Errorf("Unable to retrieve NAP from CARFAC: %v", errnum)
	}
	return resultFloats, nil
}

func (c *carfac) BM() (result []float32, err error) {
	resultAry, resultFloats := newFloatAry(c.numChannels * c.numSamples)
	if errnum := C.carfac_bm(c.cf, resultAry); errnum != 0 {
		return nil, fmt.Errorf("Unable to retrieve BM from CARFAC: %v", errnum)
	}
	return resultFloats, nil
}
