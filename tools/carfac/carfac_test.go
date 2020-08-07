package carfac

import "testing"

func TestCarfac(t *testing.T) {
	cf := New(48000)
	buf := make([]float32, cf.NumSamples())
	cf.Run(buf)
	nap, err := cf.NAP()
	if err != nil {
		t.Fatal(err)
	}
	if len(nap) != cf.NumChannels()*cf.NumSamples() {
		t.Errorf("Wanted %v samples, got %v", cf.NumChannels()*cf.NumSamples(), len(nap))
	}
}
