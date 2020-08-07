/*
 *  * Copyright 2020 Google LLC
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
 *
 */
package spectrum

import (
	"math"
	"math/cmplx"

	"github.com/mjibson/go-dsp/fft"
)

type S struct {
	SignalPower  []float64
	NoisePower   []float64
	BinBandwidth float64
	SampleRate   int
}

func (s *S) toFloat32(f []float64) []float32 {
	rval := make([]float32, len(f))
	for i := range rval {
		rval[i] = float32(f[i])
	}
	return rval
}

func (s *S) F32SignalPower() []float32 {
	return s.toFloat32(s.SignalPower)
}

func (s *S) F32NoisePower() []float32 {
	return s.toFloat32(s.NoisePower)
}

func Compute(buffer []float64, sampleRate int) S {
	coefficients := fft.FFTReal(buffer)
	halfCoefficients := len(coefficients) / 2
	invBuffer := 1 / float64(len(buffer))
	spec := S{
		BinBandwidth: float64(sampleRate) / float64(len(buffer)),
		SampleRate:   sampleRate,
	}

	spec.SignalPower = make([]float64, halfCoefficients)
	for bin := range spec.SignalPower {
		if bin == 0 || bin == halfCoefficients {
			continue
		}
		gain := cmplx.Abs(coefficients[bin]) * invBuffer * 2
		power := 0.5 * gain * gain
		spec.SignalPower[bin] = 10 * math.Log10(power)
	}
	totalMean := (cmplx.Abs(coefficients[0]) + cmplx.Abs(coefficients[halfCoefficients])) * invBuffer

	spec.NoisePower = make([]float64, halfCoefficients)
	for bin := range spec.NoisePower {
		if bin == 0 || bin == halfCoefficients {
			continue
		}
		noiseSquares := 0.0
		totalSquares := 0.0
		for otherBin, otherCoeff := range coefficients {
			square := (real(otherCoeff)*real(otherCoeff) + imag(otherCoeff)*imag(otherCoeff)) * invBuffer
			totalSquares += square
			if otherBin != bin && len(coefficients)-otherBin != bin {
				noiseSquares += square
			}
		}
		noisePower := noiseSquares*invBuffer - totalMean*totalMean
		if noisePower <= 0 {
			noisePower = 1e-10
		}
		spec.NoisePower[bin] = 10 * math.Log10(noisePower)
	}
	return spec
}
