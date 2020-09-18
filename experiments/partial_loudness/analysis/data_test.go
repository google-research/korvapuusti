package analysis

import (
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/google-research/korvapuusti/tools/synthesize/signals"

	tf "github.com/ryszard/tfutils/proto/tensorflow/core/example"
)

func TestToTFExample(t *testing.T) {
	eq := &EquivalentLoudness{
		EntryType: "1",
		Calibration: Calibration{
			HeadphoneFrequencyResponseHash: "2",
			FullScaleSineDBSPL:             3,
		},
		Run: Run{
			ID: "4",
		},
		Evaluation: Evaluation{
			ID:        "5",
			Frequency: 6,
			Probe: signals.SamplerWrapper{
				Type: "7",
				Params: map[string]interface{}{
					"Type": "8",
					"Params": map[string]interface{}{
						"Onset": map[string]interface{}{
							"Delay":    9.0,
							"Duration": 10.0,
						},
						"Level":     11.0,
						"Frequency": 12.0,
					},
				},
			},
			Combined: signals.SamplerWrapper{
				Type: "13",
				Params: map[string]interface{}{
					"Type": "14",
					"Params": map[string]interface{}{
						"Onset": map[string]interface{}{
							"Delay":    15.0,
							"Duration": 16.0,
						},
						"Level":     17.0,
						"Frequency": 18.0,
					},
				},
			},
		},
		Results: Results{
			ProbeGainForEquivalentLoudness:  19,
			ProbeDBSPLForEquivalentLoudness: 20,
		},
		Analysis: Analysis{
			BinWidth: 21,
			Rate:     22,
			Channels: [][]float64{
				[]float64{23, 24, 25},
				[]float64{26, 27, 28},
			},
			SignalLevel: [][]float64{
				[]float64{29, 30, 31},
				[]float64{32, 33, 34},
			},
			NoiseLevel: [][]float64{
				[]float64{35, 36, 37},
				[]float64{38, 39, 40},
			},
		},
	}
	ex, err := eq.ToTFExample()
	if err != nil {
		t.Fatal(err)
	}
	for k, v := range map[string]*tf.Feature{
		"EquivalentLoudness.EntryType":                                        &tf.Feature{&tf.Feature_BytesList{&tf.BytesList{[][]byte{[]byte("1")}}}},
		"EquivalentLoudness.Calibration.HeadphoneFrequencyResponseHash":       &tf.Feature{&tf.Feature_BytesList{&tf.BytesList{[][]byte{[]byte("2")}}}},
		"EquivalentLoudness.Calibration.FullScaleSineDBSPL":                   &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{3}}}},
		"EquivalentLoudness.Run.ID":                                           &tf.Feature{&tf.Feature_BytesList{&tf.BytesList{[][]byte{[]byte("4")}}}},
		"EquivalentLoudness.Evaluation.ID":                                    &tf.Feature{&tf.Feature_BytesList{&tf.BytesList{[][]byte{[]byte("5")}}}},
		"EquivalentLoudness.Evaluation.Frequency":                             &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{6}}}},
		"EquivalentLoudness.Evaluation.Probe.Type":                            &tf.Feature{&tf.Feature_BytesList{&tf.BytesList{[][]byte{[]byte("7")}}}},
		"EquivalentLoudness.Evaluation.Probe.Params.Type":                     &tf.Feature{&tf.Feature_BytesList{&tf.BytesList{[][]byte{[]byte("8")}}}},
		"EquivalentLoudness.Evaluation.Probe.Params.Params.Onset.Delay":       &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{9}}}},
		"EquivalentLoudness.Evaluation.Probe.Params.Params.Onset.Duration":    &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{10}}}},
		"EquivalentLoudness.Evaluation.Probe.Params.Params.Level":             &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{11}}}},
		"EquivalentLoudness.Evaluation.Probe.Params.Params.Frequency":         &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{12}}}},
		"EquivalentLoudness.Evaluation.Combined.Type":                         &tf.Feature{&tf.Feature_BytesList{&tf.BytesList{[][]byte{[]byte("13")}}}},
		"EquivalentLoudness.Evaluation.Combined.Params.Type":                  &tf.Feature{&tf.Feature_BytesList{&tf.BytesList{[][]byte{[]byte("14")}}}},
		"EquivalentLoudness.Evaluation.Combined.Params.Params.Onset.Delay":    &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{15}}}},
		"EquivalentLoudness.Evaluation.Combined.Params.Params.Onset.Duration": &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{16}}}},
		"EquivalentLoudness.Evaluation.Combined.Params.Params.Level":          &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{17}}}},
		"EquivalentLoudness.Evaluation.Combined.Params.Params.Frequency":      &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{18}}}},
		"EquivalentLoudness.Results.ProbeGainForEquivalentLoudness":           &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{19}}}},
		"EquivalentLoudness.Results.ProbeDBSPLForEquivalentLoudness":          &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{20}}}},
		"EquivalentLoudness.Analysis.BinWidth":                                &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{21}}}},
		"EquivalentLoudness.Analysis.Rate":                                    &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{22}}}},
		"EquivalentLoudness.Analysis.Channels[0]":                             &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{23, 24, 25}}}},
		"EquivalentLoudness.Analysis.Channels[1]":                             &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{26, 27, 28}}}},
		"EquivalentLoudness.Analysis.SignalLevel[0]":                          &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{29, 30, 31}}}},
		"EquivalentLoudness.Analysis.SignalLevel[1]":                          &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{32, 33, 34}}}},
		"EquivalentLoudness.Analysis.NoiseLevel[0]":                           &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{35, 36, 37}}}},
		"EquivalentLoudness.Analysis.NoiseLevel[1]":                           &tf.Feature{&tf.Feature_FloatList{&tf.FloatList{[]float32{38, 39, 40}}}},
	} {
		if !proto.Equal(v, ex.Features.Feature[k]) {
			t.Errorf("Got %v at %v, wanted %v", ex.Features.Feature[k], k, v)
		}
	}
}
