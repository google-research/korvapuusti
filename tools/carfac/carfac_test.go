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
