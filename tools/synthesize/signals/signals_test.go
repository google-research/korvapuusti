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
package signals

import (
	"fmt"
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
	"gonum.org/v1/gonum/stat"
)

const (
	tolerance = 0.001
)

func testFrequencyResponseFunc(f Hz) (DB, error) {
	return DB((f - 1000.0) * 0.001), nil
}

func makeSignal(frequency Hz, gain float64, rate Hz, len int) Float64Slice {
	result := Float64Slice{}
	period := rate.Period()
	for i := 0; i < len; i++ {
		result = append(result, gain*math.Sin(2*math.Pi*float64(i)*float64(period)*float64(frequency)))
	}
	return result
}

func TestToFloat32AddLevel(t *testing.T) {
	signal := makeSignal(1000, 1, 48000, 4800)
	if l := dBFS(signal); math.Abs(float64(l)-0) > tolerance {
		t.Errorf("Got level %v, wanted 0", l)
	}
	f32signal := signal.ToFloat32AddLevel(10)
	for idx := range f32signal {
		signal[idx] = float64(f32signal[idx])
	}
	if l := dBFS(signal); math.Abs(float64(l)-10) > tolerance {
		t.Errorf("Got level %v, wanted 10", l)
	}
}

func dBFS(f Float64Slice) DB {
	samples := make([]float64, len(f))
	for idx := range f {
		samples[idx] = f[idx]
	}
	// The 2 is to shift the energy to full scale sine scale.
	return DB(10 * math.Log10(2*stat.Variance(samples, nil)))
}

func TestAddLevel(t *testing.T) {
	signal := makeSignal(1000, 1, 48000, 4800)
	if l := dBFS(signal); math.Abs(float64(l)-0) > tolerance {
		t.Errorf("Got level %v, wanted 0", l)
	}
	signal.AddLevel(5)
	if l := dBFS(signal); math.Abs(float64(l)-5) > tolerance {
		t.Errorf("Got level %v, wanted 5", l)
	}
}

func TestLoadCalibrateFrequencyResponse(t *testing.T) {
	frequencyResponseData := []map[string]float64{
		{
			"100": -2,
			"200": -4,
			"300": -2,
		},
		{
			"100": -1,
			"200": -4,
			"300": -2,
		},
	}
	frequencyResponse, err := LoadCalibrateFrequencyResponse(frequencyResponseData)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		f            Hz
		wantedOffset DB
	}{
		{
			f:            100,
			wantedOffset: -1.5,
		},
		{
			f:            200,
			wantedOffset: -4,
		},
		{
			f:            300,
			wantedOffset: -2,
		},
		{
			f:            150,
			wantedOffset: -1.5/2.0 + -4.0/2.0,
		},
		{
			f:            250,
			wantedOffset: -3,
		},
		{
			f:            225,
			wantedOffset: -4.0*3.0/4.0 + -2.0/4.0,
		},
		{
			f:            275,
			wantedOffset: -4.0/4.0 + -2.0*3.0/4.0,
		},
	} {
		gotOffset, err := frequencyResponse(tc.f)
		if err != nil {
			t.Error(err)
		}
		if gotOffset != tc.wantedOffset {
			t.Errorf("Got %v for %v, wanted %v", gotOffset, tc.f, tc.wantedOffset)
		}
	}
}

func TestOnset(t *testing.T) {
	signal := Float64Slice{1, 1, 1, 1, 1, 1}
	ts := TimeStretch{0, 6}
	for _, tc := range []struct {
		signal       Float64Slice
		onset        Onset
		wantedResult Float64Slice
	}{
		{
			onset: Onset{
				Shape: Sudden,
				Delay: 2,
			},
			wantedResult: Float64Slice{0, 0, 1, 1, 1, 1},
		},
		{
			onset: Onset{
				Shape:    Linear,
				Delay:    1,
				Duration: 4,
			},
			wantedResult: Float64Slice{0, 0, 0.25, 0.5, 0.75, 1},
		},
	} {
		filterSignal := make(Float64Slice, len(signal))
		copy(filterSignal, signal)
		err := tc.onset.Filter(filterSignal, ts)
		if err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(filterSignal, tc.wantedResult); diff != "" {
			t.Errorf("%+v produced %+v, but wanted %+v", tc.onset, filterSignal, tc.wantedResult)
		}
	}
}

func TestPower(t *testing.T) {
	rate := Hz(48000.0)
	length := 100000
	for _, tc := range []struct {
		frequency   Hz
		gain        float64
		wantedPower Power
	}{
		{
			frequency:   100,
			gain:        1,
			wantedPower: 0.5,
		},
		{
			frequency:   100,
			gain:        0.5,
			wantedPower: 0.125,
		},
	} {
		signal := makeSignal(tc.frequency, tc.gain, rate, length)
		power := signal.Power()
		if math.Abs(float64(power-tc.wantedPower)) > tolerance {
			t.Errorf("got power %v, wanted %v", power, tc.wantedPower)
		}
	}
}

func TestSpectrumGains(t *testing.T) {
	rate := Hz(1000.0)
	length := 10
	for _, tc := range []struct {
		frequency   Hz
		gain        float64
		wantedGains Float64Slice
	}{
		{
			frequency:   100,
			gain:        1,
			wantedGains: Float64Slice{0, 1, 0, 0, 0},
		},
	} {
		signal := makeSignal(tc.frequency, tc.gain, rate, length)
		gains := signal.SpectrumGains()
		if diff := cmp.Diff(gains, tc.wantedGains); !gains.EqTol(tc.wantedGains, tolerance) && diff != "" {
			t.Errorf("got gains %+v, wanted %+v: %v", gains, tc.wantedGains, diff)
		}
	}
}

func TestSpectrumSignals(t *testing.T) {
	rate := Hz(1000.0)
	for _, tc := range []struct {
		frequency         Hz
		gain              float64
		wantedPeakSignals Signals
	}{
		{
			frequency: 100,
			gain:      0.5,
			wantedPeakSignals: Signals{
				{
					Shape:     Sine,
					Frequency: 100,
					Level:     -6.020599913279609,
				},
			},
		},
		{
			frequency: 130,
			gain:      0.7,
			wantedPeakSignals: Signals{
				{
					Shape:     Sine,
					Frequency: 130,
					Level:     -3.098039199714846,
				},
			},
		},
	} {
		signal := makeSignal(tc.frequency, tc.gain, rate, 1000)
		foundPeakSignals := signal.PeakSignals(rate, 0.5)
		if !foundPeakSignals.EqTol(tc.wantedPeakSignals, tolerance) {
			t.Errorf("got %+v from %v/%v, wanted %+v", foundPeakSignals, tc.frequency, tc.gain, tc.wantedPeakSignals)
		}
	}
}

func TestParse(t *testing.T) {
	for _, testCase := range []struct {
		spec         string
		wantedSignal Sampler
		wantedError  error
	}{
		{
			spec: `{"Type": "NoisyFMSignal", "Params": {"Frequency": 440, "FMFrequency": 7.5, "Level": -10, "Seed": 12}}`,
			wantedSignal: &NoisyFMSignal{
				Shape:       Sine,
				Frequency:   440,
				FMFrequency: 7.5,
				Level:       -10,
				Seed:        12,
			},
		},
		{
			spec: `{"Type": "NoisyAMSignal", "Params": {"Frequency": 440, "AMFrequency": 7.5, "Level": -10, "Seed": 12}}`,
			wantedSignal: &NoisyAMSignal{
				Shape:       Sine,
				Frequency:   440,
				AMFrequency: 7.5,
				Level:       -10,
				Seed:        12,
			},
		},
		{
			spec: `{"Type": "Noise", "Params": {"LowerLimit": 800, "UpperLimit": 1000, "Level": -15}}`,
			wantedSignal: &Noise{
				Color:      White,
				LowerLimit: 800,
				UpperLimit: 1000,
				Level:      -15,
			},
		},
		{
			spec: `{"Type": "Signal", "Params": {"Frequency": 440, "Level": -10}}`,
			wantedSignal: &Signal{
				Shape:     Sine,
				Frequency: 440,
				Level:     -10,
			},
		},
		{
			spec: `{"Type": "Signal", "Params": {"Onset": {"Delay": 0.4}, "Frequency": 1050, "Level": 20}}`,
			wantedSignal: &Signal{
				Onset: Onset{
					Shape: Sudden,
					Delay: 0.4,
				},
				Shape:     Sine,
				Frequency: 1050,
				Level:     20,
			},
		},
		{
			spec: `{"Type": "Signal", "Params": {"Onset": {"Delay": -0.4}, "Frequency": 18000}}`,
			wantedSignal: &Signal{
				Onset: Onset{
					Shape: Sudden,
					Delay: -0.4,
				},
				Shape:     Sine,
				Frequency: 18000,
				Level:     0,
			},
		},
		{
			spec: `{"Type": "Superposition", "Params": [{"Type": "Signal", "Params": {"Frequency": 2050}}]}`,
			wantedSignal: Superposition{
				&Signal{
					Frequency: 2050,
				},
			},
		},
		{
			spec:        `{"Type": "plur", "Params": {}}`,
			wantedError: fmt.Errorf(`unknown sampler type "plur"`),
		},
		{
			spec:        `..`,
			wantedError: fmt.Errorf(`invalid character '.' looking for beginning of value`),
		},
	} {
		foundSignal, foundError := ParseSampler(testCase.spec)
		if (foundError == nil) != (testCase.wantedError == nil) {
			t.Errorf("got error %v from %q, wanted %v", foundError, testCase.spec, testCase.wantedError)
		} else if foundError != nil && foundError.Error() != testCase.wantedError.Error() {
			t.Errorf("got error %v from %q, wanted %s", foundError.Error(), testCase.spec, testCase.wantedError.Error())
		}
		if diff := cmp.Diff(foundSignal, testCase.wantedSignal); diff != "" {
			t.Errorf("got %+v from %q, wanted %+v: %v", foundSignal, testCase.spec, testCase.wantedSignal, diff)
		}
	}
}

type samplerTestCase struct {
	s                     Sampler
	frequencyResponse     FrequencyResponse
	wantedPeakSignals     Signals
	wantedSpectrumGains   Float64Slice
	wantedPower           Power
	wantedError           error
	wantedSilentSeconds   float64
	forceRate             Hz
	forceTimeStretch      *TimeStretch
	forcePeakSignalsRatio float64
}

func (t samplerTestCase) verify() error {
	rate := Hz(48000.0)
	if t.forceRate != 0.0 {
		rate = t.forceRate
	}
	ts := TimeStretch{0, 1}
	if t.forceTimeStretch != nil {
		ts = *t.forceTimeStretch
	}
	period := 1.0 / rate
	signal, err := t.s.Sample(ts, rate, t.frequencyResponse)
	if err != nil {
		if t.wantedError == nil {
			return err
		} else if err.Error() == t.wantedError.Error() {
			return nil
		}
		return err
	}
	if t.wantedError != nil {
		return fmt.Errorf("got no error, wanted %+v to produce %v", t.s, t.wantedError)
	}
	if t.wantedPeakSignals != nil {
		ratio := 0.5
		if t.forcePeakSignalsRatio != 0 {
			ratio = t.forcePeakSignalsRatio
		}
		foundPeakSignals := signal.PeakSignals(rate, ratio)
		if !foundPeakSignals.EqTol(t.wantedPeakSignals, tolerance) {
			return fmt.Errorf("got peak signals %+v but wanted %+v to produce %+v", foundPeakSignals, t.s, t.wantedPeakSignals)
		}
	}
	if t.wantedSpectrumGains != nil {
		foundSpectrumGains := signal.SpectrumGains()
		if !foundSpectrumGains.EqTol(t.wantedSpectrumGains, tolerance) {
			return fmt.Errorf("got spectrum gains %+v but wanted %+v to produce %+v", foundSpectrumGains, t.s, t.wantedSpectrumGains)
		}
	}
	foundPower := signal.Power()
	if math.Abs(float64(foundPower-t.wantedPower)) > tolerance {
		return fmt.Errorf("got power %v but wanted %+v to produce %v", foundPower, t.s, t.wantedPower)
	}
	silentSeconds := 0.0
	for idx := range signal {
		if signal[idx] == 0.0 {
			silentSeconds += float64(period)
		}
	}
	if math.Abs(silentSeconds-t.wantedSilentSeconds) > tolerance {
		return fmt.Errorf("got %v silent seconds, wanted %+v to produce %v silent seconds", silentSeconds, t.s, t.wantedSilentSeconds)
	}
	return nil
}

func TestSuperposition(t *testing.T) {
	for _, tc := range []samplerTestCase{
		{
			s: Superposition{
				Signal{
					Onset: Onset{
						Shape: Sudden,
						Delay: 0.5,
					},
					Shape:     Sine,
					Frequency: 1000,
					Level:     -10,
				},
				Signal{
					Onset: Onset{
						Shape: Sudden,
						Delay: 0.3,
					},
					Shape:     Sine,
					Frequency: 2000,
					Level:     -10,
				},
			},
			wantedPeakSignals: Signals{
				{Frequency: 1000,
					Level: -16.020599913264867,
				},
				{
					Frequency: 2000,
					Level:     -13.098039199730351,
				},
			},
			wantedPower:         0.05999999999992881,
			wantedSilentSeconds: 0.3,
		},
		{
			s: Superposition{
				Signal{
					Shape:     Sine,
					Frequency: 2000,
					Level:     0,
				},
			},
			wantedPeakSignals: Signals{
				{
					Frequency: 2000,
					Level:     0.0,
				},
			},
			wantedPower: 0.5,
		},
		{
			s: Superposition{
				Signal{
					Shape:     Sine,
					Frequency: 2000,
					Level:     -7,
				},
				Signal{
					Shape:     Sine,
					Frequency: 5000,
					Level:     -7,
				},
			},
			wantedPeakSignals: Signals{
				{
					Frequency: 2000,
					Level:     -7.0,
				},
				{
					Frequency: 5000,
					Level:     -7.0,
				},
			},
			wantedPower: 0.1999,
		},
	} {
		if err := tc.verify(); err != nil {
			t.Error(err)
		}
	}
}

func TestSignal(t *testing.T) {
	for _, tc := range []samplerTestCase{
		{
			s: Signal{
				Shape:     Sine,
				Frequency: 500,
				Level:     0,
			},
			frequencyResponse: testFrequencyResponseFunc,
			wantedPeakSignals: Signals{
				{
					Frequency: 500,
					Level:     0.5,
				},
			},
			wantedPower: 0.5 * Power(math.Pow(10.0, 0.5/10.0)),
		},
		{
			s: Signal{
				Shape:     Sine,
				Frequency: 2000,
				Level:     0,
			},
			frequencyResponse: testFrequencyResponseFunc,
			wantedPeakSignals: Signals{
				{
					Frequency: 2000,
					Level:     -1.0,
				},
			},
			wantedPower: 0.5 * Power(math.Pow(10.0, -1.0/10.0)),
		},
		{
			s: Signal{
				Shape:     Sine,
				Frequency: 1000,
				Level:     0,
			},
			frequencyResponse: testFrequencyResponseFunc,
			wantedPeakSignals: Signals{
				{
					Frequency: 1000,
					Level:     0.0,
				},
			},
			wantedPower: 0.5,
		},
		{
			s: Signal{
				Shape:     Sine,
				Frequency: 1000,
				Level:     0,
			},
			wantedPeakSignals: Signals{
				{
					Frequency: 1000,
					Level:     0.0,
				},
			},
			wantedPower: 0.5,
		},
		{
			s: Signal{
				Shape:     Sine,
				Frequency: 2500,
				Level:     -6,
			},
			wantedPeakSignals: Signals{
				{
					Frequency: 2500,
					Level:     -6.0,
				},
			},
			wantedPower: Power(math.Pow(10.0, -6.0/10.0)) / 2,
		},
		{
			s: Signal{
				FM: FM{
					Width:     100,
					Frequency: 100,
				},
				Shape:     Sine,
				Frequency: 2000,
				Level:     -6,
			},
			frequencyResponse: testFrequencyResponseFunc,
			wantedPeakSignals: Signals{
				{
					Frequency: 1900,
					Level:     -14.12994793217803,
				},
				{
					Frequency: 2000,
					Level:     -9.324527032907175,
				},
				{
					Frequency: 2100,
					Level:     -14.12994793217803,
				},
			},
			wantedPower: Power(math.Pow(10.0, -7.0/10.0)) / 2,
		},
		{
			s: Signal{
				FM: FM{
					Width:     100,
					Frequency: 100,
				},
				Shape:     Sine,
				Frequency: 2500,
				Level:     -6,
			},
			wantedPeakSignals: Signals{
				{
					Frequency: 2400,
					Level:     -13.12994793217803,
				},
				{
					Frequency: 2500,
					Level:     -8.324527032907175,
				},
				{
					Frequency: 2600,
					Level:     -13.12994793217803,
				},
			},
			wantedPower: Power(math.Pow(10.0, -6.0/10.0)) / 2,
		},
		{
			s: Signal{
				AM: AM{
					Fraction:  0.5,
					Frequency: 1000,
				},
				Shape:     Sine,
				Frequency: 2000,
				Level:     -6,
			},
			frequencyResponse: testFrequencyResponseFunc,
			wantedPeakSignals: Signals{
				{
					Frequency: 1000.0,
					Level:     -12.020599913276246,
				},
				{
					Frequency: 2000.0,
					Level:     -13.020599913290981,
				},
			},
			wantedPower: 0.056339359330931656,
		},
		{
			s: Signal{
				AM: AM{
					Fraction:  0.5,
					Frequency: 1000,
				},
				Shape:     Sine,
				Frequency: 2500,
				Level:     -6,
			},
			wantedPeakSignals: Signals{
				{
					Frequency: 1000.0,
					Level:     -12.020599913276246,
				},
				{
					Frequency: 2500.0,
					Level:     -12.020599913290981,
				},
			},
			wantedPower: 0.0627971607876856,
		},
	} {
		if err := tc.verify(); err != nil {
			t.Error(err)
		}
	}
}

func TestFMNoise(t *testing.T) {
	for _, tc := range []samplerTestCase{
		{
			s: &NoisyFMSignal{
				Shape:       Sine,
				Frequency:   2000,
				FMFrequency: 7.5,
				Level:       0,
			},
			frequencyResponse: testFrequencyResponseFunc,
			wantedPeakSignals: Signals{
				{
					Frequency: 1940.0,
					Level:     -11.983525919896652,
				},
				{
					Frequency: 1985.0,
					Level:     -12.768670735492549,
				},
				{
					Frequency: 2000.0,
					Level:     -13.250035441017559,
				},
				{
					Frequency: 2015.0,
					Level:     -12.925096213872031,
				},
				{
					Frequency: 2060.0,
					Level:     -11.848607621924683,
				},
			},
			wantedPower:           0.5 * Power(math.Pow(10.0, -1.0/10.0)),
			forcePeakSignalsRatio: 0.8,
		},
		{
			s: &NoisyFMSignal{
				Shape:       Sine,
				Frequency:   2000,
				FMFrequency: 7.5,
				Level:       0,
			},
			wantedPeakSignals: Signals{
				{
					Frequency: 1940.0,
					Level:     -10.983525919896655,
				},
				{
					Frequency: 1985.0,
					Level:     -11.768670735492549,
				},
				{
					Frequency: 2000.0,
					Level:     -12.250035441017559,
				},
				{
					Frequency: 2015.0,
					Level:     -11.925096213872031,
				},
				{
					Frequency: 2060.0,
					Level:     -10.848607621924685,
				},
			},
			wantedPower:           0.5,
			forcePeakSignalsRatio: 0.8,
		},
	} {
		if err := tc.verify(); err != nil {
			t.Error(err)
		}
	}
}

func TestAMNoise(t *testing.T) {
	for _, tc := range []samplerTestCase{
		{
			s: &NoisyAMSignal{
				Shape:       Sine,
				Frequency:   2000,
				AMFrequency: 7.5,
				Level:       0,
			},
			frequencyResponse: testFrequencyResponseFunc,
			wantedPeakSignals: Signals{
				{
					Frequency: 2000.0,
					Level:     -1.04762461738375294,
				},
			},
			wantedPower: 0.5 * Power(math.Pow(10, -1.0/10.0)),
		},
		{
			s: &NoisyAMSignal{
				Shape:       Sine,
				Frequency:   2000,
				AMFrequency: 7.5,
				Level:       0,
			},
			wantedPeakSignals: Signals{
				{
					Frequency: 2000.0,
					Level:     -0.04762461738375294,
				},
			},
			wantedPower: 0.5,
		},
	} {
		if err := tc.verify(); err != nil {
			t.Error(err)
		}
	}
}

func TestNoise(t *testing.T) {
	for _, tc := range []samplerTestCase{
		{
			s: &Noise{
				LowerLimit: 0,
				UpperLimit: 250,
				Level:      0,
			},
			wantedSpectrumGains: []float64{0.19956147760195608, 0.6037301956977843, 0.7505630521497461, 0.22663055211981142, 0.14423435234451365, 1.60181777003812e-16, 2.630087941734199e-16, 1.5552972796815324e-16, 3.0603615779402827e-16, 1.3986459080436508e-16},
			wantedPower:         0.5,
			forceRate:           1000.0,
			forceTimeStretch:    &TimeStretch{0, 0.02},
		},
		{
			s: &Noise{
				LowerLimit: 0,
				UpperLimit: 250,
				Level:      0,
			},
			frequencyResponse: func(f Hz) (DB, error) {
				return DB(f / 2.5), nil
			},
			wantedSpectrumGains: []float64{0.010372902545003955, 0.003138097872746913, 0.00039013127620547206, 1.177991193043902e-05, 7.497082595746869e-07, 3.503739015785315e-18, 3.5236570605778894e-18, 1.7163516786632708e-18, 3.9422435812461906e-18, 1.7041216874547862e-18},
			wantedPower:         0.5 * Power(math.Pow(10.0, -50.0/10.0)),
			forceRate:           1000.0,
			forceTimeStretch:    &TimeStretch{0, 0.02},
		},
	} {
		if err := tc.verify(); err != nil {
			t.Error(err)
		}
	}
}
