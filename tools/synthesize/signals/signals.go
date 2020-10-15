/* Package signals contains logic to express and synthesize audio signals. *
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
package signals

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/cmplx"
	"math/rand"
	"reflect"
	"sort"
	"strconv"

	"github.com/mjibson/go-dsp/fft"
	"github.com/youpy/go-wav"
)

const (
	// FullScaleSinePower is 0.5 due to power = avg(sum(v^2)) - avg(v)^2.
	FullScaleSinePower Power = 0.5
)

// Hz is cycles per second.
type Hz float64

// Period returns the period of this frequency.
func (h Hz) Period() Seconds {
	return Seconds(1.0 / h)
}

// Power is the signal power, which is equivalent to the variance ( avg(sum(v^2)) - avg(v)^2 ) of a signal.
type Power float64

// DB returns the power converted to Decibel.
func (p Power) DB() DB {
	return DB(10 * math.Log10(float64(p)))
}

// DB is power expressed on a logarithm scale.
type DB float64

// Power returns the power of this Decibel level.
func (d DB) Power() Power {
	return Power(math.Pow(10, float64(d/10)))
}

// Gain returns the gain of this Decibel level.
func (d DB) Gain() float64 {
	return math.Pow(10, float64(d/20))
}

// Seconds is a point in time.
type Seconds float64

// TimeStretch defines a stretch of time.
type TimeStretch struct {
	// FromInclusive is the start of the stretch of time, inclusive.
	FromInclusive Seconds
	// ToExclusive is the end of the stretch of time, exclusive.
	ToExclusive Seconds
}

// Len returns the length of this time stretch.
func (t TimeStretch) Len() Seconds {
	return t.ToExclusive - t.FromInclusive
}

// FrequencyDiscrimination returns the theoretical minim difference
// between two frequencies needed for a DCT or FFT to distinguish between
// them during this time stretch.
func (t TimeStretch) FrequencyDiscrimination() Hz {
	return Hz(1.0 / t.Len())
}

// OutsideKnownFrequencyResponseError means that the asked for frequency is outside the known
// frequency range for a FrequencyResponse.
var OutsideKnownFrequencyResponseError = errors.New("Outside known frequency response")

// FrequencyResponse functions return a DB offset for a given frequency.
type FrequencyResponse func(f Hz) (DB, error)

type sortedOffset struct {
	f      Hz
	offset DB
}

type sortedOffsets []sortedOffset

func (s sortedOffsets) Len() int {
	return len(s)
}

func (s sortedOffsets) Less(i, j int) bool {
	return s[i].f < s[j].f
}

func (s sortedOffsets) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// LoadCalibrateFrequencyResponse returns a FrequencyResponse from the measurements, which
// has to be of the format produced by the tools/calibrate tool.
// The returned FrequencyResponse will linearly interpolate between the values in the
// provided measurements.
func LoadCalibrateFrequencyResponse(inputMappings []map[string]float64) (FrequencyResponse, error) {
	mergedMappings := map[Hz][]DB{}
	for _, inputMapping := range inputMappings {
		for inputFString, inputOffset := range inputMapping {
			inputF, err := strconv.ParseFloat(inputFString, 64)
			if err != nil {
				return nil, err
			}
			mergedMappings[Hz(inputF)] = append(mergedMappings[Hz(inputF)], DB(inputOffset))
		}
	}
	averagedMappings := map[Hz]DB{}
	for inputF, inputOffsets := range mergedMappings {
		sum := DB(0.0)
		for _, inputOffset := range inputOffsets {
			sum += inputOffset
		}
		averagedMappings[Hz(inputF)] = sum / DB(len(inputOffsets))
	}
	sortedMappings := sortedOffsets{}
	for inputF, averagedOffset := range averagedMappings {
		sortedMappings = append(sortedMappings, sortedOffset{f: inputF, offset: averagedOffset})
	}
	sort.Sort(sortedMappings)
	return func(f Hz) (DB, error) {
		if f < sortedMappings[0].f || f > sortedMappings[len(sortedMappings)-1].f {
			return 0.0, OutsideKnownFrequencyResponseError
		}
		index1 := sort.Search(len(sortedMappings), func(idx int) bool {
			return sortedMappings[idx].f > f
		})
		if index1 == len(sortedMappings) {
			return sortedMappings[len(sortedMappings)-1].offset, nil
		}
		f1 := sortedMappings[index1-1].f
		offset1 := sortedMappings[index1-1].offset
		f2 := sortedMappings[index1].f
		offset2 := sortedMappings[index1].offset
		partOffset2 := (f - f1) / (f2 - f1)
		partOffset1 := 1 - partOffset2
		return DB(partOffset1)*offset1 + DB(partOffset2)*offset2, nil
	}, nil
}

// Sampler can synthesize a signal for a given time.
type Sampler interface {
	// Sample returns samples during the provided time stretch at the given sample rate.
	Sample(t TimeStretch, rate Hz, speakerFrequencyResponse FrequencyResponse) (Float64Slice, error)
}

// NoiseColor defines a distribution of noise.
type NoiseColor int

const (
	// White defines a noise that has a average of zero, and where all frequencies within the limits
	// have equal gain.
	White NoiseColor = iota
)

func (n NoiseColor) String() string {
	switch n {
	case White:
		return "White"
	}
	return "Unknown"
}

// OnsetShape defines the shape of a signal onset.
type OnsetShape int

const (
	// Sudden is when the signal gets peak level between one sample and the next.
	Sudden OnsetShape = iota
	// Linear is when the signal increases linearly.
	Linear
)

func (o OnsetShape) String() string {
	switch o {
	case Sudden:
		return "Sudden"
	case Linear:
		return "Linear"
	}
	return "Unknown"
}

// Onset defines how a signal starts.
type Onset struct {
	// Shape is the shape of the ramp up.
	Shape OnsetShape
	// Delay is the delay before onset starts.
	Delay Seconds
	// Duration is how long it takes for the signal to reach peak level.
	Duration Seconds
}

// Filter filters the signal during the given time stretch with this onset.
func (o Onset) Filter(signal Float64Slice, ts TimeStretch) error {
	rate := Hz(float64(len(signal)) / float64(ts.Len()))
	period := rate.Period()
	switch o.Shape {
	case Sudden:
		t := ts.FromInclusive
		for idx := range signal {
			if t < o.Delay {
				signal[idx] = 0.0
			} else {
				break
			}
			t += period
		}
		return nil
	case Linear:
		peakT := o.Delay + o.Duration
		if ts.FromInclusive >= peakT {
			return nil
		}
		k := 1.0 / o.Duration
		t := ts.FromInclusive
		for idx := range signal {
			if t < o.Delay {
				signal[idx] = 0.0
			} else if t < peakT {
				signal[idx] *= float64(k) * float64(t-o.Delay)
			} else {
				break
			}
			t += period
		}
		return nil
	}
	return fmt.Errorf("unknown onset shape %v", o.Shape)
}

// Noise describes a band limited noise source.
type Noise struct {
	// Onset is the onset of this noise.
	Onset Onset
	// Color is the distribution of this source.
	Color NoiseColor
	// LowerLimit is the lower (inclusive) limit of this source.
	LowerLimit Hz
	// UpperLimit is the upper (exclusive) limit of this source.
	UpperLimit Hz
	// Level is the level of this signal compared to a full scale sine.
	Level DB
	// Seed is the random seed for this source.
	Seed int64
}

func (n *Noise) String() string {
	return fmt.Sprintf("%+v", *n)
}

// Sample samples this noise source at the given rate within the given time slice.
// NB: Assumes that this duration is all that is sampled, and will generate samples
// that (for this duration) should be indistinguisable from noise.
// Generates all samples at levels appropriate for the provided FrequencyResponse.
func (n *Noise) Sample(ts TimeStretch, rate Hz, speakerFrequencyResponse FrequencyResponse) (Float64Slice, error) {
	switch n.Color {
	case White:
		nSamples := int(float64(ts.Len()) * float64(rate))
		coefficients := make([]complex128, nSamples)
		freqStepHz := ts.FrequencyDiscrimination()
		fMinIdx := int(math.Round(float64(n.LowerLimit / freqStepHz)))
		fMaxIdx := int(math.Round(float64(n.UpperLimit / freqStepHz)))
		randSource := rand.NewSource(n.Seed)
		r := rand.New(randSource)
		for i := fMinIdx; i < fMaxIdx; i++ {
			frequencyResponseScaling := 1.0
			if speakerFrequencyResponse != nil {
				frequencyResponseOffset, err := speakerFrequencyResponse(freqStepHz * Hz(i))
				if err != nil {
					return nil, err
				}
				frequencyResponseScaling = math.Pow(10, float64(-frequencyResponseOffset/20.0))
			}
			coefficients[i] = complex(r.NormFloat64()*frequencyResponseScaling, r.NormFloat64()*frequencyResponseScaling)
		}
		samples := fft.IFFT(coefficients)
		result := make(Float64Slice, len(samples))
		pc := &PowerCalculator{}
		for idx := range samples {
			sample := real(samples[idx])
			result[idx] = sample
			pc.Feed(sample)
		}
		frequencyResponseOffset := DB(0.0)
		if speakerFrequencyResponse != nil {
			var err error
			frequencyResponseOffset, err = speakerFrequencyResponse(0.5 * (n.LowerLimit + n.UpperLimit))
			if err != nil {
				return nil, err
			}
		}
		result.AddLevel(FullScaleSinePower.DB() - pc.Power().DB() + n.Level - frequencyResponseOffset)
		if err := n.Onset.Filter(result, ts); err != nil {
			return nil, err
		}
		return result, nil
	}
	return nil, fmt.Errorf("unknown noise color %q, cant synthesize", n.Color)
}

// NoisyFMSignal is a signal with a phase-noisy frequency modulator.
type NoisyFMSignal struct {
	// Onset is the onset of this signal.
	Onset Onset
	// Shape is the shape of this signal.
	Shape SignalShape
	// Frequency is the average frequency of this signal.
	Frequency Hz
	// Level is the level of this signal compared to a full scale sine.
	Level DB
	// FMFrequency is the expected frequency of the frequency modulation of this signal.
	FMFrequency Hz
	// Seed is the random seed for this source.
	Seed int64
}

func (n *NoisyFMSignal) String() string {
	return fmt.Sprintf("%+v", *n)
}

// Sample samples this signal during the provided time stretch, at the provided rate.
// Generates all samples at the level appropriate for the center frequency using the provided FrequencyResponse.
func (n NoisyFMSignal) Sample(ts TimeStretch, rate Hz, speakerFrequencyResponse FrequencyResponse) (Float64Slice, error) {
	switch n.Shape {
	case Sine:
		period := rate.Period()
		result := Float64Slice{}
		fmT := ts.FromInclusive
		// The NoisyFMSignal is modulated by another sine that has a phase noise applied.
		// Empirical tests showed that this, at clearly noticeable levels, caused a
		// distinct noisy hiss - unless this phase noise was integrated more, which
		// is what randI0 and randI1 do: randI0 is integrated uniform noise, while
		// randI1 is integrated randI0, and then fmT integrates this once more.
		randSource := rand.NewSource(n.Seed)
		r := rand.New(randSource)
		randI0 := 0.5
		randI1 := 0.5
		pc := &PowerCalculator{}
		for t := ts.FromInclusive; t < ts.ToExclusive; t += period {
			// Allowing the frequency to change with 0.005 was selected after empirical listening tests.
			fmMod := float64(n.Frequency) * 0.005 * math.Sin(2*math.Pi*float64(fmT)*float64(n.FMFrequency))
			val := math.Sin(2*math.Pi*float64(t)*float64(n.Frequency) + fmMod)
			pc.Feed(val)
			result = append(result, val)
			randI0 = randI0*0.9 + 0.1*r.Float64()
			randI1 = randI1*0.9 + 0.1*randI0
			// randI0 and randI1 are both between 0 and 1, with a mean at 0.5.
			// The -0.5 centers the factor around 0, and 0.00005 is chosen to sound good - be noticeable
			// but not have a strong hiss.
			fmT += period + Seconds((randI1-0.5)*0.00005)
		}
		frequencyResponseOffset := DB(0.0)
		if speakerFrequencyResponse != nil {
			var err error
			frequencyResponseOffset, err = speakerFrequencyResponse(n.Frequency)
			if err != nil {
				return nil, err
			}
		}
		result.AddLevel(FullScaleSinePower.DB() - pc.Power().DB() + n.Level - frequencyResponseOffset)
		if err := n.Onset.Filter(result, ts); err != nil {
			return nil, err
		}
		return result, nil
	}
	return nil, fmt.Errorf("unknown signal shape %v, can't synthesize", n.Shape)
}

// NoisyAMSignal is a signal with a phase-noisy amplitude modulator.
type NoisyAMSignal struct {
	// Onset is the onset of this signal.
	Onset Onset
	// Shape is the shape of this signal.
	Shape SignalShape
	// Frequency is the average frequency of this signal.
	Frequency Hz
	// Level is the level of this signal compared to a full scale sine.
	Level DB
	// AMFrequency is the expected frequency of the amplitude modulation of this signal.
	AMFrequency Hz
	// Seed is the random seed for this source.
	Seed int64
}

func (n *NoisyAMSignal) String() string {
	return fmt.Sprintf("%+v", *n)
}

// Sample samples this signal during the provided time stretch, at the provided rate.
// Generates all samples at the level appropriate for the center frequency using the provided FrequencyResponse.
func (n NoisyAMSignal) Sample(ts TimeStretch, rate Hz, speakerFrequencyResponse FrequencyResponse) (Float64Slice, error) {
	switch n.Shape {
	case Sine:
		period := rate.Period()
		result := Float64Slice{}
		amT := ts.FromInclusive
		// The NoisyAMSignal is modulated by another sine that has a phase noise applied.
		// Empirical tests showed that this, at clearly noticeable levels, caused a
		// distinct noisy hiss - unless this phase noise was integrated more, which
		// is what randI0 and randI1 do: randI0 is integrated uniform noise, while
		// randI1 is integrated randI0, and then amT integrates this once more.
		randSource := rand.NewSource(n.Seed)
		r := rand.New(randSource)
		randI0 := 0.5
		randI1 := 0.5
		pc := &PowerCalculator{}
		for t := ts.FromInclusive; t < ts.ToExclusive; t += period {
			// 0.82 and 0.18 are selected after empirical listening tests to sound good.
			val := math.Sin(2*math.Pi*float64(t)*float64(n.Frequency)) * (0.87 + 0.13*math.Sin(2*math.Pi*float64(amT)*float64(n.AMFrequency)))
			pc.Feed(val)
			result = append(result, val)
			randI0 = randI0*0.9 + 0.1*r.Float64()
			randI1 = randI1*0.9 + 0.1*randI0
			// randI0 and randI1 are both between 0 and 1, with a mean at 0.5.
			// The -0.5 centers the factor around 0, and 0.0004 is chosen to sound good - be noticeable
			// but not have a strong hiss.
			amT += period + Seconds((randI1-0.5)*0.0004)
		}
		frequencyResponseOffset := DB(0.0)
		if speakerFrequencyResponse != nil {
			var err error
			frequencyResponseOffset, err = speakerFrequencyResponse(n.Frequency)
			if err != nil {
				return nil, err
			}
		}
		result.AddLevel(FullScaleSinePower.DB() - pc.Power().DB() + n.Level - frequencyResponseOffset)
		if err := n.Onset.Filter(result, ts); err != nil {
			return nil, err
		}
		return result, nil
	}
	return nil, fmt.Errorf("unknown signal shape %v, can't synthesize", n.Shape)
}

// SignalShape defines the known shapes of audio signals.
type SignalShape int

const (
	// Sine defines a sine wave shape.
	Sine SignalShape = iota
)

func (s SignalShape) String() string {
	switch s {
	case Sine:
		return "Sine"
	}
	return "Unknown"
}

// AM is an amptlitude modulation modulating a signal.
type AM struct {
	// Fraction is the fraction of the amplitude coming from the AM.
	Fraction float64
	// Frequency is the frequency of oscillation between low and high amplitude.
	Frequency Hz
}

// FM is a frequency modulation modulating a signal.
type FM struct {
	// Width is the distance between low and high modulation.
	Width Hz
	// Frequency is the frequency of oscillation between low and high frequency.
	Frequency Hz
}

// Signal describes a signal.
type Signal struct {
	// Onset is the onset of this signal.
	Onset Onset
	// Shape is the shape of this signal.
	Shape SignalShape
	// Frequency is the average frequency of this signal.
	Frequency Hz
	// FM is any frequency modulation applied to the signal.
	FM FM
	// AM is any amplitude modulation applied to the signal.
	AM AM
	// Level is the level of this signal compared to a full scale sine.
	Level DB
}

func (s *Signal) String() string {
	return fmt.Sprintf("%+v", *s)
}

// Sample samples this signal during the provided time stretch, at the provided rate.
// Generates all samples at the level appropriate for the center frequency using the provided FrequencyResponse.
func (s Signal) Sample(ts TimeStretch, rate Hz, speakerFrequencyResponse FrequencyResponse) (Float64Slice, error) {
	switch s.Shape {
	case Sine:
		period := rate.Period()
		result := Float64Slice{}
		gain := s.Level.Gain()
		for t := ts.FromInclusive; t < ts.ToExclusive; t += period {
			freqMod := 0.0
			if s.FM.Width > 0 && s.FM.Frequency > 0 {
				freqMod = float64(s.FM.Width/s.FM.Frequency) * math.Sin(2*math.Pi*float64(t)*float64(s.FM.Frequency))
			}
			frequencyResponseGainScale := 1.0
			if speakerFrequencyResponse != nil {
				frequencyResponseOffset, err := speakerFrequencyResponse(s.Frequency)
				if err != nil {
					return nil, err
				}
				frequencyResponseGainScale = math.Pow(10.0, float64(-frequencyResponseOffset)/20.0)
			}
			result = append(result, frequencyResponseGainScale*float64(gain)*(1.0-s.AM.Fraction)*math.Sin(2*math.Pi*float64(t)*float64(s.Frequency)+freqMod)+float64(gain)*s.AM.Fraction*math.Sin(2*math.Pi*float64(t)*float64(s.AM.Frequency)))
		}
		if err := s.Onset.Filter(result, ts); err != nil {
			return nil, err
		}
		return result, nil
	}
	return nil, fmt.Errorf("unknown signal shape %q, can't synthesize", s.Shape)
}

// EqTol returns true if the signal is equal to the other signal
// within the given tolerance.
func (s Signal) EqTol(o Signal, tol float64) bool {
	if s.Shape != o.Shape || math.Abs(float64(s.Frequency-o.Frequency)) > tol || math.Abs(float64(s.Level-o.Level)) > tol {
		return false
	}
	if s.Onset.Shape != o.Onset.Shape || math.Abs(float64(s.Onset.Delay-o.Onset.Delay)) > tol || math.Abs(float64(s.Onset.Duration-o.Onset.Duration)) > tol {
		return false
	}
	if math.Abs(float64(s.FM.Width-o.FM.Width)) > tol || math.Abs(float64(s.FM.Frequency-o.FM.Frequency)) > tol {
		return false
	}
	if math.Abs(float64(s.AM.Fraction-o.AM.Fraction)) > tol || math.Abs(float64(s.AM.Frequency-o.FM.Frequency)) > tol {
		return false
	}
	return true
}

// SamplerWrapper encodes a sampler by containing the type of sampler as a string
// along with the parameters of the underlying sampler type.
type SamplerWrapper struct {
	Type   string
	Params interface{}
}

var (
	typeMap = map[string]reflect.Type{
		reflect.TypeOf(Signal{}).Name():        reflect.TypeOf(Signal{}),
		reflect.TypeOf(Noise{}).Name():         reflect.TypeOf(Noise{}),
		reflect.TypeOf(NoisyAMSignal{}).Name(): reflect.TypeOf(NoisyAMSignal{}),
		reflect.TypeOf(NoisyFMSignal{}).Name(): reflect.TypeOf(NoisyFMSignal{}),
	}
)

// Sampler returns the sampler wrapped in the SamplerWrapper.
func (s *SamplerWrapper) Sampler() (Sampler, error) {
	if s.Type == reflect.TypeOf(Superposition{}).Name() {
		b, err := json.Marshal(s.Params)
		if err != nil {
			return nil, err
		}
		content := []SamplerWrapper{}
		if err := json.Unmarshal(b, &content); err != nil {
			return nil, fmt.Errorf("unable to decode %s as []SampleWrapper{}: %v", b, err)
		}
		super := Superposition{}
		for _, wrapper := range content {
			sampler, err := wrapper.Sampler()
			if err != nil {
				return nil, err
			}
			super = append(super, sampler)
		}
		return super, nil
	}
	template, found := typeMap[s.Type]
	if !found {
		return nil, fmt.Errorf("unknown sampler type %q", s.Type)
	}
	val := reflect.New(template)
	b, err := json.Marshal(s.Params)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, val.Interface()); err != nil {
		return nil, fmt.Errorf("unable to decode %s as %q: %v", b, s.Type, err)
	}
	return val.Interface().(Sampler), nil
}

// ParseSampler parses a spec and returns a Sampler.
func ParseSampler(spec string) (Sampler, error) {
	js := &SamplerWrapper{}
	if err := json.Unmarshal([]byte(spec), js); err != nil {
		return nil, err
	}
	return js.Sampler()
}

// Signals is a slice of signals.
type Signals []Signal

// EqTol returns true if the signals are equal to the other signals
// within the given tolerance.
func (s Signals) EqTol(o Signals, tol float64) bool {
	if len(s) != len(o) {
		return false
	}
	for idx := range s {
		if !s[idx].EqTol(o[idx], tol) {
			return false
		}
	}
	return true
}

// Superposition is a superposition of signals.
type Superposition []Sampler

func (s Superposition) String() string {
	return fmt.Sprintf("%+v", []Sampler(s))
}

// Sample samples the superposition of these signals at time t seconds.
func (s Superposition) Sample(ts TimeStretch, rate Hz, speakerFrequencyResponse FrequencyResponse) (Float64Slice, error) {
	period := rate.Period()
	t := ts.FromInclusive
	var result Float64Slice
	for samplerIdx, sampler := range s {
		sampled, err := sampler.Sample(ts, rate, speakerFrequencyResponse)
		if err != nil {
			return nil, err
		}
		if result == nil {
			result = make([]float64, len(sampled))
		} else if len(result) != len(sampled) {
			return nil, fmt.Errorf("sampler %v of %+v returned a different number of samples (%v) than the ones before (%v)", samplerIdx, s, len(sampled), len(result))
		}
		for sampleIdx := range sampled {
			result[sampleIdx] += sampled[sampleIdx]
		}
		t += period
	}
	return result, nil
}

// Float64Slice represents a sound buffer of floats between -1 and 1.
type Float64Slice []float64

// EqTol returns whether the other float slice is equal to this one,
// within the given tolerance.
func (f Float64Slice) EqTol(o Float64Slice, tol float64) bool {
	if len(f) != len(o) {
		return false
	}
	for idx := range f {
		if math.Abs(f[idx]-o[idx]) > tol {
			return false
		}
	}
	return true
}

// WriteWAV writes the samples as a WAV file to a writer, declaring a given
// sample rate. Assumes the slice contains only values between -1.0 and 1.0.
func (f Float64Slice) WriteWAV(w io.Writer, rate float64) error {
	wavSamples := make([]wav.Sample, len(f))
	for idx := range f {
		val := int(f[idx] * float64(math.MaxInt16))
		wavSamples[idx] = wav.Sample{
			Values: [2]int{val, val},
		}
	}
	buf := &bytes.Buffer{}
	wavWriter := wav.NewWriter(buf, uint32(len(f)), 2, uint32(rate), 16)
	if err := wavWriter.WriteSamples(wavSamples); err != nil {
		return err
	}
	_, err := io.Copy(w, buf)
	return err
}

// SpectrumGains returns a slice with the gain (the complex absolute value) of the
// first half of the FFT of the slice.
func (f Float64Slice) SpectrumGains() Float64Slice {
	coefficients := fft.FFTReal(f)
	halfCoefficients := len(coefficients) / 2
	invBuffer := 1 / float64(len(f))
	gains := make(Float64Slice, halfCoefficients)
	for bin := range gains {
		gains[bin] = cmplx.Abs(coefficients[bin]) * invBuffer * 2
	}
	return gains
}

// PowerCalculator calculates power of signals.
type PowerCalculator struct {
	sum          float64
	sumOfSquares float64
	len          float64
}

// Feed feeds the calculator the next sample.
func (p *PowerCalculator) Feed(f float64) {
	p.sum += f
	p.sumOfSquares += f * f
	p.len++
}

// Power returns the power of the signal so far.
func (p *PowerCalculator) Power() Power {
	mean := p.sum / p.len
	return Power(p.sumOfSquares/p.len - mean*mean)
}

// Power returns the signal power of the slice.
func (f Float64Slice) Power() Power {
	pc := &PowerCalculator{}
	for _, val := range f {
		pc.Feed(val)
	}
	return pc.Power()
}

// PeakSignals returns a slice of sine signals that describe
// the center frequencies of peaks in frequency space of this slice
// where the peak reaches over ratio of the max value across the entire
// frequency space.
func (f Float64Slice) PeakSignals(rate Hz, ratio float64) Signals {
	gains := f.SpectrumGains()
	maxGain := 0.0
	for bin := 1; bin < len(gains); bin++ {
		if gains[bin] > maxGain {
			maxGain = gains[bin]
		}
	}
	cutoff := maxGain * ratio
	buffer := [3]float64{}
	binBW := float64(rate) / float64(len(f))
	signals := Signals{}
	for bin := 1; bin < len(gains); bin++ {
		buffer[0], buffer[1] = buffer[1], buffer[2]
		buffer[2] = gains[bin]
		if buffer[0] < buffer[1] && buffer[1] > buffer[2] {
			if buffer[1] > cutoff {
				signals = append(signals, Signal{
					Shape:     Sine,
					Frequency: Hz(binBW * float64(bin-1)),
					Level:     DB(20.0 * math.Log10(buffer[1])),
				})
			}
		}
	}
	return signals
}

// SetDBFS set the level of f to be the given DB away from a full scale sine.
func (f Float64Slice) SetDBFS(d DB) {
	f.AddLevel(FullScaleSinePower.DB() - f.Power().DB() + d)
}

// AddLevel adds a number of Decibel to the signal.
func (f Float64Slice) AddLevel(d DB) {
	scale := math.Pow(10, float64(d)/20.0)
	for idx := range f {
		f[idx] *= scale
	}
}

// ToFloat32AddLevel returns a signal of float32's with the provided level added.
func (f Float64Slice) ToFloat32AddLevel(d DB) []float32 {
	scale := math.Pow(10, float64(d)/20.0)
	f32slice := make([]float32, len(f))
	for idx := range f {
		f32slice[idx] = float32(scale) * float32(f[idx])
	}
	return f32slice
}
