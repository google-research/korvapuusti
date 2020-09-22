/* data contains definitions of data formats for the partial_loudness
 * experiment.
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
package analysis

import (
	"fmt"
	"reflect"

	"github.com/google-research/korvapuusti/tools/spectrum"
	"github.com/google-research/korvapuusti/tools/synthesize/signals"

	tf "github.com/ryszard/tfutils/proto/tensorflow/core/example"
)

// Calibration defines the calibration of the equipment before the evaluation.
type Calibration struct {
	// HeadphoneFrequencyResponseHash is a hash of the headphone frequency response definition used to produce calibrated sounds.
	HeadphoneFrequencyResponseHash string
	// FullScaleSineDBSPL identifies the dB SPL level used to calibrate a full scale sine (with gain 1.0).
	FullScaleSineDBSPL float64
}

// Run identifies this run as a part of a series of evaluations.
type Run struct {
	// ID is the unique ID for this run.
	ID string
}

// Evaluation identifies this particular evaluation, which frequency in the run was evaluated, and the exact parameters used to produce the sounds.
type Evaluation struct {
	// ID is the unique ID for this evaluation.
	ID string
	// Frequency is the frequency used for this evaluation as a part of the run.
	Frequency float64
	// Probe describes the probe sound played in the evaluation.
	Probe signals.SamplerWrapper
	// Combined describes the combined sound played in the evaluation.
	Combined signals.SamplerWrapper
}

// Results defines the results of the human evaluation.
type Results struct {
	// ProbeGainForEquivalentLoudness identifies the gain used when the evaluator found the levels to be equally loud.
	ProbeGainForEquivalentLoudness float64
	// ProbeDBSPLForEquivalentLoudness identifies the db SPL (calculated using the Level in Evaluation.Probe,
	// Calibration.FullScaleSineDBSPL and ProbeGainForEquivalentLoudness) when the evaluator found the levels to be equally loud.
	ProbeDBSPLForEquivalentLoudness float64
}

// Analysis contains the CARFAC analysis of the sound played to the evaluator, without using the headphone frequency response calibration
// (since CARFAC gets the raw audio, and doesn't need to be calibrated for headphones).
type Analysis struct {
	// Channels[channelIDX][sampleStep] is the time domain output of the CARFAC channels.
	Channels [][]float64
	// ChannelPoles[channelIDX is the pole frequency for each channel.
	ChannelPoles []float64
	// ChannelSpectrums is the results of FFT of the Channels.
	ChannelSpectrums []spectrum.S
}

// EquivalentLoudness describes an evaluation and its results.
type EquivalentLoudness struct {
	// EntryType is used to filter out actual EquivalentLoudness events from frequency response events in the log.
	EntryType string
	// Calibration describes the calibration used during the evaluation.
	Calibration Calibration
	// Run identifies the evaluation as part of a run.
	Run Run
	// Evaluation defines the sounds evaluated.
	Evaluation Evaluation
	// Results defines the results of the human evaluation.
	Results Results
	// Analysis contains automatied analysis of the sounds evaluated.
	Analysis Analysis
}

func (e *EquivalentLoudness) toTFExample(val reflect.Value, namePrefix string, ex *tf.Example) error {
	typ := val.Type()
	switch typ.Kind() {
	case reflect.String:
		ex.Features.Feature[namePrefix] = &tf.Feature{&tf.Feature_BytesList{&tf.BytesList{[][]byte{[]byte(val.String())}}}}
	case reflect.Float64:
		ex.Features.Feature[namePrefix] = &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{float32(val.Float())}}}}
	case reflect.Slice:
		elemTyp := typ.Elem()
		switch elemTyp.Kind() {
		case reflect.Struct:
			fallthrough
		case reflect.Slice:
			for elemIdx := 0; elemIdx < val.Len(); elemIdx++ {
				if err := e.toTFExample(val.Index(elemIdx), fmt.Sprintf("%v[%v]", namePrefix, elemIdx), ex); err != nil {
					return err
				}
			}
		case reflect.Float64:
			floats := make([]float32, val.Len())
			for idx := 0; idx < val.Len(); idx++ {
				floats[idx] = float32(val.Index(idx).Float())
			}
			ex.Features.Feature[namePrefix] = &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{floats}}}
		case reflect.Interface:
			for elemIdx := 0; elemIdx < val.Len(); elemIdx++ {
				if err := e.toTFExample(val.Index(elemIdx).Elem(), fmt.Sprintf("%v[%v]", namePrefix, elemIdx), ex); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("%v %v is of an invalid slice type %v", namePrefix, val.Interface(), typ)
		}
	case reflect.Map:
		iter := val.MapRange()
		for iter.Next() {
			if err := e.toTFExample(iter.Value(), namePrefix+"."+fmt.Sprint(iter.Key()), ex); err != nil {
				return err
			}
		}
	case reflect.Struct:
		for fieldIdx := 0; fieldIdx < typ.NumField(); fieldIdx++ {
			fieldTyp := typ.Field(fieldIdx)
			if fieldTyp.Tag.Get("proto") != "-" {
				fieldVal := val.Field(fieldIdx)
				if err := e.toTFExample(fieldVal, namePrefix+"."+fieldTyp.Name, ex); err != nil {
					return err
				}
			}
		}
	case reflect.Interface:
		if err := e.toTFExample(val.Elem(), namePrefix, ex); err != nil {
			return err
		}
	default:
		return fmt.Errorf("%v %v is of an invalid type %v", namePrefix, val.Interface(), typ)
	}
	return nil
}

// ToTFExample converts an EquivalentLoudness to a tf.Example for compact logging.
func (e *EquivalentLoudness) ToTFExample() (*tf.Example, error) {
	ex := &tf.Example{
		Features: &tf.Features{
			Feature: map[string]*tf.Feature{},
		},
	}
	if err := e.toTFExample(reflect.ValueOf(*e), "EquivalentLoudness", ex); err != nil {
		return nil, err
	}
	return ex, nil
}
