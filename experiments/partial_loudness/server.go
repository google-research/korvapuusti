/* partial_loudness is a package that lets you run a server presenting a web page
 * that plays sounds and lets you adjust levels.
 *
 * Run it and browse to http://localhost:12000.
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
//go:generate go-bindata -o bindata/bindata.go -pkg bindata images/ html/ js/
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"text/template"

	"github.com/google-research/korvapuusti/experiments/partial_loudness/bindata"
	"github.com/google-research/korvapuusti/tools/synthesize/signals"
)

const (
	rate = 48000
)

func MustAsset(n string) []byte {
	b, err := bindata.Asset(n)
	if err != nil {
		panic(err)
	}
	return b
}

var (
	signalRequestReg = regexp.MustCompile("^/signal/(.*)\\.((probe|mask|calib)(\\.[0-9_]+)?\\.wav)$")
	indexTemplate    = template.Must(template.New("index.html").Parse(string(MustAsset("html/index.html"))))
)

var (
	experimentOutput = flag.String("experiment_output",
		filepath.Join(os.Getenv("HOME"), "partial_loudness_output"),
		"Path to store the experiment results to.")
	listen                         = flag.String("listen", "localhost:12000", "Interface and port to listen for connections on.")
	erbWidth                       = flag.Float64("erb_width", 0.0, "Preset ERB width for white noise in the experiment.")
	maskFrequencies                = flag.String("mask_frequencies", "", "Preset mask frequencies for the experiment.")
	maskLevels                     = flag.String("mask_levels", "", "Preset mask levels for the experiment.")
	probeLevel                     = flag.Float64("probe_level", 0.0, "Preset probe level for the experiment.")
	erbApart                       = flag.Float64("erb_apart", 0.0, "Preset ERB apart for the experiment.")
	exactFrequencies               = flag.String("exact_frequencies", "", "Preset exact frequencies to present the probe at.")
	signalType                     = flag.String("signal_type", "", "Preset signal type for the experiment.")
	hideControls                   = flag.Bool("hide_controls", false, "Whether to hide the controls in the experiment.")
	headphoneFrequencyResponseFile = flag.String("headphone_frequency_response_file", "", "Frequency response file for headphones used, produced by the calibrate/calibrate.html tool.")
)

type server struct {
	headphoneFrequencyResponse signals.FrequencyResponse
}

func (s *server) renderIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	if err := indexTemplate.Execute(w, map[string]interface{}{
		"ExperimentOutput":               *experimentOutput,
		"ERBWidth":                       *erbWidth,
		"MaskFrequencies":                *maskFrequencies,
		"MaskLevels":                     *maskLevels,
		"ProbeLevel":                     *probeLevel,
		"ERBApart":                       *erbApart,
		"SignalType":                     *signalType,
		"HideControls":                   *hideControls,
		"ExactFrequencies":               *exactFrequencies,
		"HeadphoneFrequencyResponseFile": *headphoneFrequencyResponseFile,
	}); err != nil {
		s.handleError(w, err)
		return
	}
}

func (s *server) handleError(w http.ResponseWriter, err error) {
	log.Print(err)
	http.Error(w, err.Error(), http.StatusInternalServerError)
}

func (s *server) renderSignal(w http.ResponseWriter, r *http.Request) {
	match := signalRequestReg.FindStringSubmatch(r.URL.Path)
	if match == nil {
		s.handleError(w, fmt.Errorf("missing signal spec in path %q", r.URL.Path))
		return
	}
	escaped, err := url.QueryUnescape(match[1])
	if err != nil {
		s.handleError(w, fmt.Errorf("unable to unescape signal spec %q", match[1]))
		return
	}
	wrapper := signals.SamplerWrapper{}
	if err := json.Unmarshal([]byte(escaped), &wrapper); err != nil {
		s.handleError(w, err)
		return
	}
	signal, err := wrapper.Sampler()
	if err != nil {
		s.handleError(w, err)
		return
	}
	samples, err := signal.Sample(signals.TimeStretch{FromInclusive: 0, ToExclusive: 5}, rate, s.headphoneFrequencyResponse)
	if err != nil {
		s.handleError(w, fmt.Errorf("Unable to sample %+v: %v", signal, err))
		return
	}
	w.Header().Set("Content-Type", "audio/wav")
	if err := samples.WriteWAV(w, rate); err != nil {
		s.handleError(w, fmt.Errorf("Unable to render WAV response: %v", err))
		return
	}
	return
}

type equivalentLoudness struct {
	EntryType                       string
	RunID                           string
	EvaluationID                    string
	FullScaleSineDBSPL              float64
	ProbeGainForEquivalentLoudness  float64
	ProbeDBSPLForEquivalentLoudness float64
	ERBWidth                        float64
	MaskFrequencies                 []float64
	MaskLevels                      []float64
	ProbeFrequency                  float64
	ProbeLevel                      float64
	SignalType                      string
	ERBApart                        float64
}

func (s *server) log(i interface{}) error {
	filePath := filepath.Join(*experimentOutput, "evaluations.json")
	if err := os.MkdirAll(*experimentOutput, 0755); err != nil {
		return err
	}
	logFile, err := os.OpenFile(filePath,
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer logFile.Close()
	encoder := json.NewEncoder(logFile)
	return encoder.Encode(i)
}

func (s *server) logEquivalentLoudness(w http.ResponseWriter, r *http.Request) {
	filePath := filepath.Join(*experimentOutput, "evaluations.json")
	if r.Method == "POST" {
		equiv := &equivalentLoudness{}
		if err := json.NewDecoder(r.Body).Decode(equiv); err != nil {
			s.handleError(w, err)
			return
		}
		equiv.EntryType = "EquivalentLoudnessMeasurement"
		val := reflect.ValueOf(*equiv)
		for fieldNo := 0; fieldNo < val.NumField(); fieldNo++ {
			if val.Type().Field(fieldNo).Name != "EntryType" && val.Field(fieldNo).IsZero() {
				s.handleError(w, fmt.Errorf("field %v of %+v is zero", val.Type().Field(fieldNo).Name, equiv))
				return
			}
		}
		if equiv.FullScaleSineDBSPL == 0 || equiv.ProbeGainForEquivalentLoudness == 0 || equiv.ProbeDBSPLForEquivalentLoudness == 0 {
			s.handleError(w, fmt.Errorf("%+v isn't fully populated", equiv))
			return
		}
		if err := s.log(equiv); err != nil {
			s.handleError(w, err)
			return
		}
	} else if r.Method == "GET" {
		w.Header().Set("Content-Type", "application/json")
		logFile, err := os.Open(filePath)
		if err != nil {
			if os.IsNotExist(err) {
				return
			}
			s.handleError(w, err)
			return
		}
		defer logFile.Close()
		if _, err := io.Copy(w, logFile); err != nil {
			s.handleError(w, err)
			return
		}
	} else if r.Method == "DELETE" {
		data, err := ioutil.ReadFile(filePath)
		if err != nil {
			s.handleError(w, err)
			return
		}
		lines := strings.Split(string(data), "\n")
		for strings.TrimSpace(lines[len(lines)-1]) == "" {
			lines = lines[:len(lines)-1]
		}
		text := strings.Join(lines[:len(lines)-1], "\n") + "\n"
		if err := ioutil.WriteFile(filePath, []byte(text), 0644); err != nil {
			s.handleError(w, err)
			return
		}
	}
}

func (s *server) createAssetFunc(dir string, contentType string) func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		b, err := bindata.Asset(filepath.Join(dir, filepath.Base(r.URL.Path)))
		if err != nil {
			s.handleError(w, fmt.Errorf("Unable to find asset for %q: %v", r.URL.Path, err))
			return
		}
		w.Header().Set("Content-Type", contentType)
		if _, err := io.Copy(w, bytes.NewBuffer(b)); err != nil {
			s.handleError(w, err)
			return
		}
	}
}

func main() {
	flag.Parse()
	s := &server{}
	if *headphoneFrequencyResponseFile != "" {
		blob, err := ioutil.ReadFile(*headphoneFrequencyResponseFile)
		if err != nil {
			panic(err)
		}
		measurements := []map[string]float64{}
		if err := json.Unmarshal(blob, &measurements); err != nil {
			panic(err)
		}
		freqResp, err := signals.LoadCalibrateFrequencyResponse(measurements)
		if err != nil {
			panic(err)
		}
		s.headphoneFrequencyResponse = freqResp
		s.log(map[string]interface{}{
			"EntryType":    "FrequencyResponseMeasurements",
			"Measurements": measurements,
		})
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/signal/", s.renderSignal)
	mux.HandleFunc("/log", s.logEquivalentLoudness)
	mux.HandleFunc("/images/", s.createAssetFunc("images", "image/png"))
	mux.HandleFunc("/js/", s.createAssetFunc("js", "application/javascript"))
	mux.HandleFunc("/", s.renderIndex)
	log.Printf("Starting server. Browse to http://%v", *listen)
	log.Fatal(http.ListenAndServe(*listen, mux))
}
