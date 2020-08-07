/* partial_loudness is a package that lets you run a server presenting a web page
 * that lets you play different sounds, adjust their levels, and find the
 * masking levels of concurrent sounds.
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
package main

import (
	"fmt"
	"log"
	"net/http"
	"regexp"
	"strings"

	"github.com/google-research/korvapuusti/tools/synthesize/signals"
)

var (
	signalRequestReg = regexp.MustCompile("^/signal/(.*)\\.wav$")
)

func renderIndex(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, `
<html>
<head>
<script>
function ready() {
	const probeAudio = document.getElementById("probe-audio");
	const maskAudio = document.getElementById("mask-audio");
	const levelOutput = document.getElementById("level-output");
	const probeSpec = document.getElementById("probe-spec");
	const maskSpec = document.getElementById("mask-spec");
	const probeLink = document.getElementById("probe-link");
	const maskLink = document.getElementById("mask-link");
	const raiseProbe = document.getElementById("raise-probe");
	const lowerProbe = document.getElementById("lower-probe");
	const playProbe = document.getElementById("play-probe");
	const playMask = document.getElementById("play-mask");

	let probeLevel = -30;
	probeAudio.volume = Math.pow(10.0, probeLevel / 20.0);
	levelOutput.innerText = probeLevel;

	const url = new URL(document.location.href);
	const urlProbeSpec = url.searchParams.get("probe-spec");
	if (urlProbeSpec) {
		probeSpec.value = urlProbeSpec;
		probeLink.setAttribute("href", "/signal/" + urlProbeSpec + ".wav");
		probeAudio.setAttribute("src", "/signal/" + urlProbeSpec + ".wav");
	}
	const urlMaskSpec = url.searchParams.get("mask-spec");
	if (urlMaskSpec) {
		maskSpec.value = urlMaskSpec;
		maskLink.setAttribute("href", "/signal/" + urlMaskSpec + ".wav");
		maskAudio.setAttribute("src", "/signal/" + urlMaskSpec + ".wav");
	}
	
	let focusCounter = 0;
	probeSpec.addEventListener("focus", ev => {
		focusCounter++;
	});
	maskSpec.addEventListener("focus", ev => {
		focusCounter++;
	});
	probeSpec.addEventListener("blur", ev => {
		focusCounter--;
		probeAudio.setAttribute("src", "/signal/" + ev.target.value + ".wav");
		probeLink.setAttribute("href", "/signal/" + ev.target.value + ".wav");
		const url = new URL(document.location.href);
		url.searchParams.set("probe-spec", ev.target.value);
		history.pushState(null, null, url.toString());
	});
	maskSpec.addEventListener("blur", ev => {
		focusCounter--;
		maskAudio.setAttribute("src", "/signal/" + ev.target.value + ".wav");
		maskLink.setAttribute("href", "/signal/" + ev.target.value + ".wav");
		const url = new URL(document.location.href);
		url.searchParams.set("mask-spec", ev.target.value);
		history.pushState(null, null, url.toString());
	});
	
	const raiseProbeFunc = _ => {
		if (probeLevel + 1 < 1) {
			probeLevel++;
			probeAudio.volume = Math.pow(10.0, probeLevel / 20.0);
			levelOutput.innerText = probeLevel;
		}
	};
	const lowerProbeFunc = _ => {
		if (probeLevel - 1 > -120) {
			probeLevel--;
			probeAudio.volume = Math.pow(10.0, probeLevel / 20.0);
			levelOutput.innerText = probeLevel;
		}
	};
	const playProbeFunc = _ => {
		if (probeAudio.paused) {
			probeAudio.currentTime = 0.0;
			probeAudio.play();
			maskAudio.pause();
		} else {
			probeAudio.pause();
		}
	};
	const playMaskFunc = _ => {
		if (maskAudio.paused) {
			maskAudio.currentTime = 0.0;
			maskAudio.play();
			probeAudio.pause();
		} else {
			maskAudio.pause();
		}
	};
	
	raiseProbe.addEventListener("click", raiseProbeFunc);
	lowerProbe.addEventListener("click", lowerProbeFunc);
	playProbe.addEventListener("click", playProbeFunc);
	playMask.addEventListener("click", playMaskFunc);
	
	document.addEventListener('keydown', ev => {
		if (focusCounter > 0) {
			return;
		}
		switch (ev.key) {
			case 'ArrowUp':
				ev.preventDefault();
				ev.stopPropagation();
				raiseProbeFunc();
				break;
			case 'ArrowDown':
				ev.preventDefault();
				ev.stopPropagation();
				lowerProbeFunc();
				break;
			case 'ArrowLeft':
				ev.preventDefault();
				ev.stopPropagation();
				playProbeFunc();
				break;
			case 'ArrowRight':
				ev.preventDefault();
				ev.stopPropagation();
				playMaskFunc();
				break;
		}
	});
}
</script>
</head>
<body>
<audio loop style="display:none;" src="/signal/sudden:0:0:sine:440:0.wav" id="probe-audio"></audio>
<audio loop style="display:none;" src="/signal/sudden:0:0:sine:220:-10,sudden:0.5:0:sine:440:-20.wav" id="mask-audio"></audio>
<p>
Populate 'Probe spec' and 'Mask spec' with comma separated signal specs (see parser source for details).
</p>
<div>
<label for="probe">Probe spec</label>
<input style="width:80em;" type="text" name="probe" id="probe-spec" placeholder="sudden:0:0:sine:440:0">
<a id="probe-link" href="/signal/0:sine:440:0.wav">
	<button type="submit">Download probe</button>
</a>
</div>
<div>
<label for="mask">Mask spec</label>
<input style="width:80em;" type="text" name="mask" id="mask-spec" placeholder="sudden:0:0:sine:220:-10,sudden:0.5:0:sine:440:-20">
<a id="mask-link" href="/signal/0:sine:220:-10,0.5:sine:440:-20.wav">
	<button type="submit">Download mask</button>
</a>
</div>
<div style="margin-top:5em;">
<div>
<button id="raise-probe">Raise probe [⬆]</button>
</div>
<div id="level-output">
</div>
<div>
<button id="lower-probe">Lower probe [⬇]</button>
</div>
<button id="play-probe">Play/pause probe [⬅]</button>
<button id="play-mask">Play/pause mask [➡]</button>
</div>
<script>
ready();
</script>
</body>
</html>
`)
}

func renderSignal(w http.ResponseWriter, r *http.Request) {
	match := signalRequestReg.FindStringSubmatch(r.URL.Path)
	if match == nil {
		log.Printf("Invalid signal path %q", r.URL.Path)
		http.Error(w, "Invalid signal path", http.StatusBadRequest)
		return
	}
	samplerSpecs := strings.Split(match[1], ",")
	if len(samplerSpecs) == 0 {
		log.Printf("Missing signal spec in %q", r.URL.Path)
		http.Error(w, "Missing signal spec", http.StatusBadRequest)
		return
	}
	super := make(signals.Superposition, len(samplerSpecs))
	for idx, spec := range samplerSpecs {
		sampler, err := signals.ParseSampler(spec)
		if err != nil {
			log.Printf("Invalid signal spec %q: %v", spec, err)
			http.Error(w, fmt.Sprintf("Invalid signal spec %q: %v", spec, err), http.StatusBadRequest)
			return
		}
		super[idx] = sampler
	}
	samples, err := super.Sample(signals.TimeStretch{FromInclusive: 0, ToExclusive: 5}, 48000)
	if err != nil {
		log.Printf("Unable to sample %+v: %v", super, err)
		http.Error(w, fmt.Sprintf("Unable to sample %+v: %v", super, err), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "audio/wav")
	if err := samples.WriteWAV(w, 48000); err != nil {
		log.Printf("Unable to render WAV: %v", err)
		http.Error(w, fmt.Sprintf("Unable to render WAV: %v", err), http.StatusInternalServerError)
		return
	}
	return
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/signal/", renderSignal)
	mux.HandleFunc("/", renderIndex)
	log.Printf("Starting server. Browse to http://localhost:12000")
	log.Fatal(http.ListenAndServe(":12000", mux))
}
