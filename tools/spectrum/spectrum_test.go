package spectrum

import (
	"math"
	"testing"

	"github.com/google-research/korvapuusti/tools/synthesize/signals"
)

func genSig(freq float64, amp float64, num int, rate float64) []float64 {
	res := make([]float64, num)
	period := 1.0 / rate
	for step := range res {
		res[step] = amp * math.Sin(2*math.Pi*freq*period*float64(step))
	}
	return res
}

func TestSpectrum(t *testing.T) {
	s := Compute(genSig(1, 1, 8, 8), 8)
	if s.BinWidth != 1 {
		t.Errorf("Got %+v, wanted BinWidth 1", s)
	}
	if s.Rate != 8 {
		t.Errorf("Got %+v, wanted Rate 8", s)
	}
	peakSNRFreq := signals.Hz(-1.0)
	peakSNR := -100000.0
	for idx := range s.SignalPower {
		snr := s.SignalPower[idx] - s.NoisePower[idx]
		if snr > peakSNR {
			peakSNR = snr
			peakSNRFreq = signals.Hz(idx) * s.BinWidth
		}
	}
	if peakSNRFreq != 1.0 {
		t.Errorf("Got peak SNR at %vHz, wanted 1Hz", peakSNRFreq)
	}
}