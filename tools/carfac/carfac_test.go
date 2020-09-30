package carfac

import "testing"

func TestCarfac(t *testing.T) {
	zero := 0.0
	for _, params := range []CARFACParams{
		{
			SampleRate: 48000,
			VOffset:    nil,
		},
		{
			SampleRate: 24000,
			VOffset:    &zero,
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
