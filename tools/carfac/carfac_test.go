/*
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
package carfac

import (
	"math"
	"reflect"
	"testing"
)

func makeSignal(l int) []float32 {
	buf := make([]float32, l)
	f := 1000.0
	ti := 0.0
	step := 1.0 / 48000.0
	for idx := range buf {
		buf[idx] = float32(math.Sin(2 * math.Pi * f * ti * step))
		ti += step
	}
	return buf
}

func TestDefaultParams(t *testing.T) {
	cp := CARFACParams{}
	cp.Default(44100)
	assertEq32(t, *cp.VelocityScale, 0.1)
	assertEq32(t, *cp.VOffset, 0.04)
	assertEq32(t, *cp.MinZeta, 0.1)
	assertEq32(t, *cp.MaxZeta, 0.35)
	assertEq32(t, *cp.ZeroRatio, math.Sqrt(2.0))
	assertEq32(t, *cp.HighFDampingCompression, 0.5)
	assertEq32(t, *cp.ERBPerStep, 0.5)
	assertEq32(t, *cp.ERBBreakFreq, 165.3)
	assertEq32(t, *cp.ERBQ, 1000/(24.7*4.37))
	assertEq32(t, *cp.TauLPF, 0.000080)
	assertEq32(t, *cp.Tau1Out, 0.0005)
	assertEq32(t, *cp.Tau1In, 0.01)
	assertEq32(t, *cp.ACCornerHz, 20.0)
	assertEq32(t, *cp.StageGain, 2.0)
	assertEq32(t, *cp.AGC1Scale0, 1.0)
	assertEq32(t, *cp.AGC1ScaleMul, math.Sqrt(2.0))
	assertEq32(t, *cp.AGC2Scale0, 1.65)
	assertEq32(t, *cp.AGC2ScaleMul, math.Sqrt(2.0))
	assertEq32(t, *cp.TimeConstant0, 0.002)
	assertEq32(t, *cp.TimeConstantMul, 4.0)
	assertEq32(t, *cp.AGCMixCoeff, 0.5)
}

func assertEq32(t *testing.T, f1 float64, f2 float64) {
	if float32(f1) != float32(f2) {
		t.Errorf("Got %v, wanted %v", f1, f2)
	}
}

func TestCARFAC(t *testing.T) {
	for _, tc := range []struct {
		f1        func() ([]float32, error)
		f2        func() ([]float32, error)
		wantEqual bool
		desc      string
	}{
		{
			desc: "Deterministic",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			wantEqual: true,
		},
		{
			desc: "OpenLoopDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.RunOpen(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "ERBPerStepDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				one := 1.0
				cf := New(CARFACParams{SampleRate: 48000, ERBPerStep: &one})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "VOffsetDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				zero := 0.0
				cf := New(CARFACParams{SampleRate: 48000, VOffset: &zero})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "MaxZetaDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				half := 0.5
				cf := New(CARFACParams{SampleRate: 48000, MaxZeta: &half})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "ZeroRatioDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				half := 0.5
				cf := New(CARFACParams{SampleRate: 48000, ZeroRatio: &half})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "StageGainDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				three := 3.0
				cf := New(CARFACParams{SampleRate: 48000, StageGain: &three})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "VelocityScaleDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				point2 := 0.2
				cf := New(CARFACParams{SampleRate: 48000, VelocityScale: &point2})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "HighFDampingCompressionDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				point6 := 0.6
				cf := New(CARFACParams{SampleRate: 48000, HighFDampingCompression: &point6})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "AGC1Scale0Different",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				two := 2.0
				cf := New(CARFACParams{SampleRate: 48000, AGC1Scale0: &two})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "AGC1ScaleMulDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				two := 2.0
				cf := New(CARFACParams{SampleRate: 48000, AGC1ScaleMul: &two})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "AGC2Scale0Different",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				two := 2.0
				cf := New(CARFACParams{SampleRate: 48000, AGC2Scale0: &two})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "AGC2ScaleMulDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				two := 2.0
				cf := New(CARFACParams{SampleRate: 48000, AGC2ScaleMul: &two})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "TimeConstant0Different",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				two := 2.0
				cf := New(CARFACParams{SampleRate: 48000, TimeConstant0: &two})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "TimeConstantMulDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				two := 2.0
				cf := New(CARFACParams{SampleRate: 48000, TimeConstantMul: &two})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "ERBBreakFreqDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				two_hundred := 200.0
				cf := New(CARFACParams{SampleRate: 48000, ERBBreakFreq: &two_hundred})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "ERBQDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				x := 1200 / (24.7 * 4.37)
				cf := New(CARFACParams{SampleRate: 48000, ERBQ: &x})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
		},
		{
			desc: "TauLPFDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.NAP()
			},
			f2: func() ([]float32, error) {
				x := 0.00009
				cf := New(CARFACParams{SampleRate: 48000, TauLPF: &x})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.NAP()
			},
		},
		{
			desc: "Tau1OutDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.NAP()
			},
			f2: func() ([]float32, error) {
				x := 0.006
				cf := New(CARFACParams{SampleRate: 48000, Tau1Out: &x})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.NAP()
			},
		},
		{
			desc: "Tau1InDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.NAP()
			},
			f2: func() ([]float32, error) {
				x := 0.02
				cf := New(CARFACParams{SampleRate: 48000, Tau1In: &x})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.NAP()
			},
		},
		{
			desc: "ACCornerHzDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.NAP()
			},
			f2: func() ([]float32, error) {
				x := 0.25
				cf := New(CARFACParams{SampleRate: 48000, ACCornerHz: &x})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.NAP()
			},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			res1, err := tc.f1()
			if err != nil {
				t.Fatal(err)
			}
			res2, err := tc.f2()
			if err != nil {
				t.Fatal(err)
			}
			if wasEqual := reflect.DeepEqual(res1, res2); tc.wantEqual != wasEqual {
				t.Errorf("Wanted equal to be %v, was %v", tc.wantEqual, wasEqual)
			}
		})
	}
}
