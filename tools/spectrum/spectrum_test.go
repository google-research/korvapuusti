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

func TestNoisePower(t *testing.T) {
	sampleRate := 1000.0
	sigs := []struct {
		freq               float64
		gain               float64
		expectedNoisePower float64
	}{
		{
			freq: 100,
			gain: 1,
		},
		{
			freq: 300,
			gain: 2,
		},
		{
			freq: 50,
			gain: 0.5,
		},
	}
	superpos := make([]float64, int(sampleRate))
	for outerIdx := range sigs {
		sumOfSquares := 0.0
		sum := 0.0
		for i := range superpos {
			superpos[i] += sigs[outerIdx].gain * math.Sin(math.Pi*2*sigs[outerIdx].freq*float64(i)/sampleRate)
			noiseVal := 0.0
			for innerIdx := range sigs {
				if innerIdx != outerIdx {
					noiseVal += sigs[innerIdx].gain * math.Sin(math.Pi*2*sigs[innerIdx].freq*float64(i)/sampleRate)
				}
			}
			sumOfSquares += noiseVal * noiseVal
			sum += noiseVal
		}
		avg := sum / sampleRate
		sigs[outerIdx].expectedNoisePower = sumOfSquares/sampleRate - avg*avg
	}
	spec := Compute(superpos, signals.Hz(sampleRate))
	if spec.BinWidth != 1.0 {
		t.Errorf("got bin width %v, wanted 1.0", spec.BinWidth)
	}
	for _, sig := range sigs {
		if got := float64(spec.NoisePower[int(sig.freq/float64(spec.BinWidth))]); math.Abs(got-10*math.Log10(sig.expectedNoisePower)) > 1e-9 {
			t.Errorf("got noise power %v, wanted %v", math.Pow(10, got/10), sig.expectedNoisePower)
		}
	}
}

func TestSignalPower(t *testing.T) {
	sampleRate := 1000.0
	for _, tc := range []struct {
		freq        float64
		gain        float64
		wantedPower float64
	}{
		{
			freq: 100,
			gain: 1,
		},
		{
			freq: 300,
			gain: 2,
		},
		{
			freq: 50,
			gain: 0.5,
		},
	} {
		sig := make([]float64, int(sampleRate))
		sumOfSquares := 0.0
		sum := 0.0
		for i := range sig {
			sig[i] = tc.gain * math.Sin(math.Pi*2*tc.freq*float64(i)/sampleRate)
			sumOfSquares += sig[i] * sig[i]
			sum += sig[i]
		}
		avg := sum / sampleRate
		power := sumOfSquares/sampleRate - avg*avg
		spec := ComputeSignalPower(sig, signals.Hz(sampleRate))
		if spec.BinWidth != 1.0 {
			t.Errorf("got bin width %v, wanted 1.0", spec.BinWidth)
		}
		if got := float64(spec.SignalPower[int(tc.freq/float64(spec.BinWidth))]); math.Abs(got-10*math.Log10(power)) > 1e-9 {
			t.Errorf("got power %v, wanted %v", math.Pow(10, got/10), power)
		}
	}
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
		snr := float64(s.SignalPower[idx] - s.NoisePower[idx])
		if snr > peakSNR {
			peakSNR = snr
			peakSNRFreq = signals.Hz(idx) * s.BinWidth
		}
	}
	if peakSNRFreq != 1.0 {
		t.Errorf("Got peak SNR at %vHz, wanted 1Hz", peakSNRFreq)
	}
}
