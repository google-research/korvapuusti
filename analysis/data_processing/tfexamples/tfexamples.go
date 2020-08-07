/*
Package tfexamples contains functionality for reading .wav files and performing CARFAC, FFT, and SNR calculations.

When calling exported function WriteExamplesToRecordio each .wav file in a directory gets read and on a particular
window the CARFAC features are calculated. On each of the CARFAC channels a FFT + SNR calculation is done.
These features are then saved in TFExamples in RecordIO files.

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
 *
*/
package tfexamples

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/youpy/go-wav"

	"github.com/google-research/korvapuusti/analysis/data_processing/spectrum"
	"github.com/google-research/korvapuusti/tools/carfac"
)

// Counts the number of digits in an integer.
func iterativeDigitsCount(number int64) int64 {
	count := 0
	for number != 0 {
		number /= 10
		count++
	}
	return int64(count)

}

// Extracts integer from a string (e.g., "0060.00" becomes 60)
func getNumberFromStr(inputStr string) (float32, error) {
	re := regexp.MustCompile(`[-]?\d[\d,]*[\.]?[\d{2}]*`)
	number, err := strconv.ParseFloat(re.FindString(inputStr), 32)
	if err != nil {
		return 0, err
	}
	return float32(number), nil
}

// Extracts frequencies and levels from file string
// r000001_combined_00060.00Hz_-50.00dBFS+04800.00Hz_-60.00dBFS.wav
// becomes frequencies 60, 4800 and levels -50, -60
// TODO: calculate other db level.
func extractMetadataFile(file string) ([]float32, []float32, error) {
	s := strings.Split(file, "/")
	s1 := s[len(s)-1]
	s2 := strings.Split(s1, "+")
	toneLevelProbe := strings.Split(s2[0], "_")
	toneLevelMasker := strings.Split(s2[1], "_")
	frequencyProbe, err := getNumberFromStr(toneLevelProbe[2])
	if err != nil {
		return nil, nil, err
	}
	frequencyMasker, err := getNumberFromStr(toneLevelMasker[0])
	if err != nil {
		return nil, nil, err
	}
	levelMasker, err := getNumberFromStr(toneLevelMasker[1])
	if err != nil {
		return nil, nil, err
	}
	levelProbe, err := getNumberFromStr(toneLevelProbe[3])
	if err != nil {
		return nil, nil, err
	}
	frequencies := []float32{frequencyProbe, frequencyMasker}
	levels := []float32{levelProbe, levelMasker}
	return frequencies, levels, nil
}

// Opens wave file and returns the samples in an array of max length maxSamples
func readWavfile(filePath string, maxSamples int, skipNSamples int) ([]float32, error) {
	// OpenRead the file
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	reader := wav.NewReader(f)
	defer f.Close()

	var buffer []float32
	totalNSamples := 0
	for {
		samples, err := reader.ReadSamples()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		for _, sample := range samples {
			if totalNSamples > skipNSamples {
				buffer = append(buffer, float32(reader.FloatValue(sample, 0)))
			}
			totalNSamples++
		}
		if len(buffer) >= maxSamples {
			break
		}
	}
	return buffer, nil
}

// Calculates the CARFAC NAP features from a buffer.
func getCarfacFeatures(sampleRate int, carfacBuffer []float32) (int, []float32, error) {
	cf := carfac.New(sampleRate)
	cf.Run(carfacBuffer)
	NAP, err := cf.NAP()
	if err != nil {
		return 0, nil, err
	}
	return cf.NumChannels(), NAP, nil
}

// Calculate FFT and SNR on each CARFAC channel.
func getSignalNoiseFeatures(sampleRate int, fftWindowSize int, numChannels int,
	carfacFeatures []float32, carfacStableIdx int) (int, []float32, []float32) {
	signalPowerFeatures := make([]float32, numChannels*(fftWindowSize/2))
	noisePowerFeatures := make([]float32, numChannels*(fftWindowSize/2))
	numBins := 0
	for channelIdx := 0; channelIdx < numChannels; channelIdx++ {
		buffer := make([]float64, fftWindowSize)
		for freqIdx := range buffer {
			buffer[freqIdx] = float64(carfacFeatures[(channelIdx + (carfacStableIdx+freqIdx)*numChannels)])
		}
		spec := spectrum.Compute(buffer, sampleRate)
		numBins = len(spec.SignalPower)
		for freqIdx := range spec.SignalPower {
			signalPowerFeatures[channelIdx*numBins+freqIdx] = float32(spec.SignalPower[freqIdx])
			noisePowerFeatures[channelIdx*numBins+freqIdx] = float32(spec.NoisePower[freqIdx])
		}
	}
	return numBins, signalPowerFeatures, noisePowerFeatures
}

type example struct {
	SampleRate     int       `json:"samplerate"`
	Frequencies    []float32 `json:"frequencies"`
	SPLs           []float32 `json:"spls"`
	ProbeFrequency float32   `json:"probefrequency"`
	ProbeLevel     int       `json:"probelevel"`
	ProbeLoudness  int       `json:"probeloudness"`
	Channels       int       `json:"channels"`
	Bins           int       `json:"bins"`
	SpectrumSignal []float32 `json:"spectrumsignal"`
	SpectrumNoise  []float32 `json:"spectrumnoise"`
}

type ExampleId struct {
	ProbeFrequency       float32 `json:"probe_frequency"`
	ProbeLevel           int     `json:"probe_level"`
	PerceivedProbeLevels []int   `json:"perceived_probe_levels"`
	WorkerIds            []int   `json:"worker_ids"`
	MaskerFrequency      float32 `json:"masker_frequency"`
	MaskerLevel          int     `json:"masker_level"`
	WavfileIdentifier    string  `json:"wavfile_identifier"`
}

// WriteExamplesToRecordIO writes TF examples to recordio shards.
func WriteExamplesToRecordIO(waveFileDirectory string, dataPath string, filePrefix string, shardSize int, carfacStableIdx int,
	fftWindowSize int, sampleRate int, skipSeconds float64, unityDBLevel float64) error {
	count := 0
	var done func()
	defer func() {
		if done != nil {
			done()
		}
	}()
	files, err := os.Open(waveFileDirectory)
	if err != nil {
		return err
	}
	readFiles, err := files.Readdir(-1)
	files.Close()
	if err != nil {
		return err
	}

	file, err := os.Open(dataPath)
	if err != nil {
		return err
	}

	defer file.Close()

	bytesData, err := ioutil.ReadAll(file)
	if err != nil {
		return err
	}

	var exampleIds []ExampleId

	error := json.Unmarshal(bytesData, &exampleIds)
	if error != nil {
		return error
	}

	// Calculate how many samples to skip.
	skipNSamples := int(float64(sampleRate) * skipSeconds)

	allData := []*example{}
	dataExamples := 0
	wroteExamples := 0

	// Loop over all files in the provided directory.
	for _, exampleID := range exampleIds {
		var foundMatch bool

		perceivedProbeLevels := exampleID.PerceivedProbeLevels
		sum := 0
		for _, perceivedLevel := range perceivedProbeLevels {
			sum = sum + perceivedLevel
		}
		meanPerceivedLevel := sum / len(perceivedProbeLevels)

		for fileIdx, infilePath := range readFiles {
			matched, _ := filepath.Match("*"+exampleID.WavfileIdentifier+".wav",
				infilePath.Name())
			// Only take combined tones.
			if matched {
				foundMatch = true
				fmt.Printf("Working on file %d: %s\n", fileIdx, infilePath.Name())

				// Extract all information needed from the file.
				frequencies, levels, err := extractMetadataFile(infilePath.Name())
				otherFrequencies := []float32{exampleID.MaskerFrequency, exampleID.ProbeFrequency}
				for idx, frequency := range frequencies {
					if frequency != otherFrequencies[idx] {
						fmt.Printf("Frequency 1: %f Frequency 2: %f", frequency, otherFrequencies[idx])
						err := errors.New("mismatch between dataFile and WavFiles")
						return err
					}
				}

				if err != nil {
					return err
				}
				buffer, err := readWavfile(path.Join(waveFileDirectory, infilePath.Name()), carfacStableIdx+(fftWindowSize*10), skipNSamples)
				if err != nil {
					return err
				}

				// Get CARFAC features.
				numChannels, NAP, err := getCarfacFeatures(sampleRate, buffer)
				if err != nil {
					return err
				}
				numBins, signalPowerFeatures, noisePowerFeatures := getSignalNoiseFeatures(sampleRate, fftWindowSize,
					numChannels, NAP, carfacStableIdx)

				realLevels := []float32{}
				for _, level := range levels {
					realLevel := level + float32(unityDBLevel)
					realLevels = append(realLevels, realLevel)
				}
				otherLevels := []float32{float32(exampleID.MaskerLevel), float32(exampleID.ProbeLevel)}
				for idx, level := range realLevels {
					fmt.Printf("level 1: %f level 2: %f", level, otherLevels[idx])
					if level != otherLevels[idx] {
						err := errors.New("mismatch between dataFile and WavFiles")
						return err
					}
				}

				// Save to TF Example and write to file.
				example := &example{
					SampleRate:     sampleRate,
					Frequencies:    frequencies,
					SPLs:           realLevels,
					ProbeFrequency: exampleID.ProbeFrequency,
					ProbeLevel:     exampleID.ProbeLevel,
					ProbeLoudness:  meanPerceivedLevel,
					Channels:       numChannels,
					Bins:           numBins,
					SpectrumSignal: signalPowerFeatures,
					SpectrumNoise:  noisePowerFeatures,
				}

				allData = append(allData, example)
				count++
				dataExamples++

				fmt.Printf("Samples total: (L) %d\n", len(buffer))
				fmt.Printf("Mean perceived level %d\n", meanPerceivedLevel)
				fmt.Printf("CARFAC features total: (L) %d\n", len(NAP))
				fmt.Printf("Signal features total: %d\n", len(signalPowerFeatures))
				fmt.Printf("Noise features total: %d\n", len(noisePowerFeatures))
				if count%shardSize == 0 {
					indentedJsonString, err := json.MarshalIndent(allData, "", "\t")
					if err != nil {
						return err
					}
					ioutil.WriteFile(fmt.Sprintf("%s_%07d-%07d.json", filePrefix, count-shardSize, count),
						indentedJsonString, os.ModePerm)
					wroteExamples += len(allData)
					allData = nil
				}
			}
		}
		if !foundMatch {
			errorString := fmt.Sprintf("Did not find a match in the wavefiledirectory for %s", exampleID.WavfileIdentifier)
			err := errors.New(errorString)
			return err
		}
	}
	indentedJSONString, err := json.MarshalIndent(allData, "", "\t")
	if err != nil {
		return err
	}
	if allData != nil {
		ioutil.WriteFile(fmt.Sprintf("%s_%07d-%07d.json", filePrefix, count-len(allData), count), indentedJSONString, os.ModePerm)
		wroteExamples += len(allData)
	}
	if wroteExamples != dataExamples {
		errorString := fmt.Sprintf("wrote %d examples but read %d examples", wroteExamples, dataExamples)
		err := errors.New(errorString)
		return err
	}
	return nil
}
