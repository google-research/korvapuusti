/* spectrum contains functions analysing spectrum composition of signals.
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
package spectrum

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"math/cmplx"

	"github.com/google-research/korvapuusti/tools/synthesize/signals"
	"github.com/mjibson/go-dsp/fft"
)

type S struct {
	Coeffs      []complex128 `json:"-" proto:"-"`
	SignalPower []signals.DB
	NoisePower  []signals.DB
	BinWidth    signals.Hz
	Rate        signals.Hz
}

func (s *S) Print(width int, w io.Writer) {
	headers := []string{}
	gains := []float64{}
	maxHeaderLen := 0
	maxGain := -math.MaxFloat64
	for i := 0; i < len(s.Coeffs)/2; i++ {
		header := fmt.Sprintf("%.2fHz ", signals.Hz(i)*s.BinWidth)
		if len(header) > maxHeaderLen {
			maxHeaderLen = len(header)
		}
		headers = append(headers, header)
		gain := cmplx.Abs(s.Coeffs[i])
		if !math.IsInf(gain, 1) && gain > maxGain {
			maxGain = cmplx.Abs(s.Coeffs[i])
		}
		gains = append(gains, gain)
	}
	gainLen := width - maxHeaderLen
	widthPerGain := float64(gainLen) / maxGain
	for i := 0; i < len(s.Coeffs)/2; i++ {
		header := bytes.NewBufferString(headers[i])
		for len(header.String()) < maxHeaderLen {
			fmt.Fprint(header, " ")
		}
		gainPart := &bytes.Buffer{}
		if math.IsInf(gains[i], 1) {
			for len(gainPart.String()) < gainLen {
				fmt.Fprintf(gainPart, "+Inf ")
			}
		} else {
			for len(gainPart.String()) < int((gains[i])*widthPerGain) {
				fmt.Fprintf(gainPart, "*")
			}
		}
		fmt.Fprintf(w, "%v%v\n", header.String(), gainPart.String())
	}
}

func (s *S) toFloat32(f []signals.DB) []float32 {
	rval := make([]float32, len(f))
	for i := range rval {
		rval[i] = float32(f[i])
	}
	return rval
}

func (s *S) Gains() []float64 {
	invBuffer := 1.0 / float64(len(s.Coeffs))
	res := make([]float64, len(s.Coeffs))
	for idx := range s.Coeffs {
		res[idx] = cmplx.Abs(s.Coeffs[idx]) * invBuffer * 2
	}
	return res
}

func (s *S) F32SignalPower() []float32 {
	return s.toFloat32(s.SignalPower)
}

func (s *S) F32NoisePower() []float32 {
	return s.toFloat32(s.NoisePower)
}

func ComputeSignalPower(buffer signals.Float64Slice, rate signals.Hz) *S {
	spec := &S{
		BinWidth: rate / signals.Hz(len(buffer)),
		Rate:     rate,
		Coeffs:   fft.FFTReal(buffer),
	}
	halfCoefficients := len(spec.Coeffs) / 2
	invBuffer := 1.0 / float64(len(buffer))

	spec.SignalPower = make([]signals.DB, halfCoefficients)
	for bin := range spec.SignalPower {
		if bin == 0 {
			continue
		}
		gain := cmplx.Abs(spec.Coeffs[bin]) * invBuffer * 2
		power := 0.5 * gain * gain
		spec.SignalPower[bin] = signals.DB(10 * math.Log10(power))
	}
	return spec
}

func Compute(buffer signals.Float64Slice, rate signals.Hz) *S {
	spec := ComputeSignalPower(buffer, rate)

	halfCoefficients := len(spec.Coeffs) / 2
	invBuffer := 1.0 / float64(len(buffer))
	totalMean := (cmplx.Abs(spec.Coeffs[0]) + cmplx.Abs(spec.Coeffs[halfCoefficients])) * invBuffer

	totalSquares := 0.0
	noiseSquares := make([]float64, len(spec.Coeffs))
	for bin, coeff := range spec.Coeffs {
		square := (real(coeff)*real(coeff) + imag(coeff)*imag(coeff)) * invBuffer
		totalSquares += square
		noiseSquares[bin] = square
	}
	spec.NoisePower = make([]signals.DB, halfCoefficients)
	for bin := range spec.NoisePower {
		if bin == 0 {
			continue
		}
		noisePower := (totalSquares-noiseSquares[bin]-noiseSquares[len(spec.Coeffs)-bin])*invBuffer - totalMean*totalMean
		if noisePower <= 0 {
			noisePower = 1e-20
		}
		spec.NoisePower[bin] = signals.DB(10 * math.Log10(noisePower))
	}
	return spec
}
