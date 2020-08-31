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
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
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
	listen           = flag.String("listen", "localhost:12000", "Interface and port to listen for connections on.")
	erbWidth         = flag.Float64("erb_width", 0.0, "Preset ERB width for white noise in the experiment.")
	maskFrequencies  = flag.String("mask_frequencies", "", "Preset mask frequencies for the experiment.")
	maskLevels       = flag.String("mask_levels", "", "Preset mask levels for the experiment.")
	probeLevel       = flag.Float64("probe_level", 0.0, "Preset probe level for the experiment.")
	erbApart         = flag.Float64("erb_apart", 0.0, "Preset ERB apart for the experiment.")
	exactFrequencies = flag.String("exact_frequencies", "", "Preset exact frequencies to present the probe at.")
	signalType       = flag.String("signal_type", "", "Preset signal type for the experiment.")
	hideControls     = flag.Bool("hide_controls", false, "Whether to hide the controls in the experiment.")
)

func renderIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	if err := indexTemplate.Execute(w, map[string]interface{}{
		"ExperimentOutput": *experimentOutput,
		"ERBWidth":         *erbWidth,
		"MaskFrequencies":  *maskFrequencies,
		"MaskLevels":       *maskLevels,
		"ProbeLevel":       *probeLevel,
		"ERBApart":         *erbApart,
		"SignalType":       *signalType,
		"HideControls":     *hideControls,
		"ExactFrequencies": *exactFrequencies,
	}); err != nil {
		handleError(w, err)
		return
	}
}

func handleError(w http.ResponseWriter, err error) {
	log.Print(err)
	http.Error(w, err.Error(), http.StatusInternalServerError)
}

func renderSignal(w http.ResponseWriter, r *http.Request) {
	match := signalRequestReg.FindStringSubmatch(r.URL.Path)
	if match == nil {
		handleError(w, fmt.Errorf("missing signal spec in path %q", r.URL.Path))
		return
	}
	wantedPath := match[2]
	escaped, err := url.QueryUnescape(match[1])
	if err != nil {
		handleError(w, fmt.Errorf("unable to unescape signal spec %q", match[1]))
		return
	}
	samplers := signals.SamplerWrappers{}
	if err := json.Unmarshal([]byte(escaped), &samplers); err != nil {
		handleError(w, err)
		return
	}
	super, err := samplers.Superposition()
	if err != nil {
		handleError(w, err)
		return
	}
	samples, err := super.Sample(signals.TimeStretch{FromInclusive: 0, ToExclusive: 5}, rate)
	if err != nil {
		handleError(w, fmt.Errorf("Unable to sample %+v: %v", super, err))
		return
	}
	outputPath := filepath.Join(*experimentOutput, "signal", wantedPath)
	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		handleError(w, fmt.Errorf("Unable to create directory %q for signals: %v", filepath.Dir(outputPath), err))
		return
	}
	outputFile, err := os.Create(outputPath)
	if err != nil {
		handleError(w, fmt.Errorf("Unable to create signal file %q: %v", outputPath, err))
		return
	}
	defer outputFile.Close()
	if err := samples.WriteWAV(outputFile, rate); err != nil {
		handleError(w, fmt.Errorf("Unable to save WAV file: %v", err))
		return
	}
	w.Header().Set("Content-Type", "audio/wav")
	if err := samples.WriteWAV(w, rate); err != nil {
		handleError(w, fmt.Errorf("Unable to render WAV response: %v", err))
		return
	}
	return
}

type equivalentLoudness struct {
	RunID                           string
	EvaluationID                    string
	ProbePath                       string
	CombinedPath                    string
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

func logEquivalentLoudness(w http.ResponseWriter, r *http.Request) {
	if r.Method == "POST" {
		equiv := &equivalentLoudness{}
		if err := json.NewDecoder(r.Body).Decode(equiv); err != nil {
			handleError(w, err)
			return
		}
		val := reflect.ValueOf(*equiv)
		for fieldNo := 0; fieldNo < val.NumField(); fieldNo++ {
			if val.Field(fieldNo).IsZero() {
				handleError(w, fmt.Errorf("field %v of %+v is zero", val.Type().Field(fieldNo).Name, equiv))
				return
			}
		}
		if equiv.ProbePath == "" || equiv.CombinedPath == "" || equiv.FullScaleSineDBSPL == 0 || equiv.ProbeGainForEquivalentLoudness == 0 || equiv.ProbeDBSPLForEquivalentLoudness == 0 {
			handleError(w, fmt.Errorf("%+v isn't fully populated", equiv))
			return
		}
		if err := os.MkdirAll(*experimentOutput, 0755); err != nil {
			handleError(w, err)
			return
		}
		logFile, err := os.OpenFile(filepath.Join(*experimentOutput, "evaluations.json"),
			os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			handleError(w, err)
			return
		}
		defer logFile.Close()
		encoder := json.NewEncoder(logFile)
		if err := encoder.Encode(equiv); err != nil {
			handleError(w, err)
			return
		}
	} else if r.Method == "GET" {
		w.Header().Set("Content-Type", "application/json")
		logFile, err := os.Open(filepath.Join(*experimentOutput, "evaluations.json"))
		if err != nil {
			if os.IsNotExist(err) {
				return
			}
			handleError(w, err)
			return
		}
		defer logFile.Close()
		if _, err := io.Copy(w, logFile); err != nil {
			handleError(w, err)
			return
		}
	}
}

func createAssetFunc(dir string, contentType string) func(w http.ResponseWriter, r *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		b, err := bindata.Asset(filepath.Join(dir, filepath.Base(r.URL.Path)))
		if err != nil {
			handleError(w, fmt.Errorf("Unable to find asset for %q: %v", r.URL.Path, err))
			return
		}
		w.Header().Set("Content-Type", contentType)
		if _, err := io.Copy(w, bytes.NewBuffer(b)); err != nil {
			handleError(w, err)
			return
		}
	}
}

func main() {
	flag.Parse()
	mux := http.NewServeMux()
	mux.HandleFunc("/signal/", renderSignal)
	mux.HandleFunc("/log", logEquivalentLoudness)
	mux.HandleFunc("/images/", createAssetFunc("images", "image/png"))
	mux.HandleFunc("/js/", createAssetFunc("js", "application/javascript"))
	mux.HandleFunc("/", renderIndex)
	log.Printf("Starting server. Browse to http://%v", *listen)
	log.Fatal(http.ListenAndServe(*listen, mux))
}
