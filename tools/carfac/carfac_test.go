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
			desc: "DhDgRatioDifferent",
			f1: func() ([]float32, error) {
				cf := New(CARFACParams{SampleRate: 48000})
				cf.Run(makeSignal(cf.NumSamples()))
				return cf.BM()
			},
			f2: func() ([]float32, error) {
				dHdGRatio := 0.1
				cf := New(CARFACParams{SampleRate: 48000, DhDgRatio: &dHdGRatio})
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
