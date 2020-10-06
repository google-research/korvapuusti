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

func TestDeterminism(t *testing.T) {
	cf1 := New(CARFACParams{SampleRate: 48000, VOffset: nil})
	cf2 := New(CARFACParams{SampleRate: 48000, VOffset: nil})
	buf := makeSignal(cf1.NumSamples())
	cf1.Run(buf)
	cf2.Run(buf)
	bm1, err := cf1.BM()
	if err != nil {
		t.Fatal(err)
	}
	bm2, err := cf2.BM()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(bm1, bm2) {
		t.Errorf("The BM output of two default CARFAC instances are not the same?")
	}
}

func TestOpenLoopMakesADifference(t *testing.T) {
	regularCF := New(CARFACParams{SampleRate: 48000, OpenLoop: false})
	openLoopCF := New(CARFACParams{SampleRate: 48000, OpenLoop: true})
	buf := makeSignal(regularCF.NumSamples())
	regularCF.Run(buf)
	openLoopCF.Run(buf)
	regularBM, err := regularCF.BM()
	if err != nil {
		t.Fatal(err)
	}
	openLoopBM, err := openLoopCF.BM()
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(regularBM, openLoopBM) {
		t.Errorf("The BM output of a default CARFAC instance and one with open loop are the same?")
	}
}

func TestERBPerStepMakesADifference(t *testing.T) {
	regularCF := New(CARFACParams{SampleRate: 48000, ERBPerStep: nil})
	one := 1.0
	oneERBPerStepCF := New(CARFACParams{SampleRate: 48000, ERBPerStep: &one})
	buf := makeSignal(regularCF.NumSamples())
	regularCF.Run(buf)
	oneERBPerStepCF.Run(buf)
	regularBM, err := regularCF.BM()
	if err != nil {
		t.Fatal(err)
	}
	oneERBPerStepBM, err := oneERBPerStepCF.BM()
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(regularBM, oneERBPerStepBM) {
		t.Errorf("The BM output of a default CARFAC instance and one with ERBPerStep set to 1.0 are the same?")
	}
	if len(regularBM) < len(oneERBPerStepBM) {
		t.Errorf("The BM output of a default CARFAC instance (with 0.5 ERB per step) is shorter than the BM output of a CARFAC instance with ERBPerStep set to 1.0?")
	}
}

func TestVOffsetMakesADifference(t *testing.T) {
	regularCF := New(CARFACParams{SampleRate: 48000, VOffset: nil})
	zero := 0.0
	zeroVOffsetCF := New(CARFACParams{SampleRate: 48000, VOffset: &zero})
	buf := makeSignal(regularCF.NumSamples())
	regularCF.Run(buf)
	zeroVOffsetCF.Run(buf)
	regularBM, err := regularCF.BM()
	if err != nil {
		t.Fatal(err)
	}
	zeroVOffsetBM, err := zeroVOffsetCF.BM()
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(regularBM, zeroVOffsetBM) {
		t.Errorf("The BM output of a default CARFAC instance and one with v_offset set to 0.0 are the same?")
	}
}

func TestCarfac(t *testing.T) {
	zero := 0.0
	one := 1.0
	for _, params := range []CARFACParams{
		{
			SampleRate: 48000,
			VOffset:    nil,
		},
		{
			SampleRate: 24000,
			VOffset:    &zero,
		},
		{
			SampleRate: 24000,
			ERBPerStep: &one,
		},
	} {
		cf := New(params)
		buf := make([]float32, cf.NumSamples())
		cf.Run(buf)
		nap, err := cf.NAP()
		if err != nil {
			t.Fatal(err)
		}
		if len(nap) != cf.NumChannels()*cf.NumSamples() {
			t.Errorf("Wanted %v samples, got %v", cf.NumChannels()*cf.NumSamples(), len(nap))
		}
		bm, err := cf.BM()
		if err != nil {
			t.Fatal(err)
		}
		if len(bm) != cf.NumChannels()*cf.NumSamples() {
			t.Errorf("Wanted %v samples, got %v", cf.NumChannels()*cf.NumSamples(), len(nap))
		}
	}
}
