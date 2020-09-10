"use strict";

// See https://en.wikipedia.org/wiki/Equivalent_rectangular_bandwidth
function erbWidthAtHz(f) {
	return 24.7 * (4.37 * f * 0.001 + 1);
}
function hZToERB(f) {
	return 21.4 * Math.log10(1 + 0.00437 * f);
}
// Simply an inversion of HzToERB.
function erbToHz(erb) {
	return (Math.pow(10, erb / 21.4) - 1) / 0.00437;
}

// Default values, and stopping the key listening on focus.
let ignoreKeyEvents = false;
function configureInput(input, externalValue, defaultValue) {
	input.addEventListener("focus", (_) => {
		ignoreKeyEvents = true;
	});
	input.addEventListener("blur", (_) => {
		ignoreKeyEvents = false;
	});
	if (externalValue) {
		input.value = externalValue;
		input.disabled = true;
	} else {
		input.value = defaultValue;
	}
}

// After document is loaded.
function documentLoaded() {
	// Don't accept some commands.
	let active = false;

	// We don't want to play louder than this to avoid hurting our ears.
	const maxLevel = 90.0; // dB SPL

	// We need to calibrate to louder than the max level, since
	// generated noise randomly produces very high or low values.
	const fullScaleSineLevel = 100.0; // dB SPL

	// The generated probe tone, that's scaled using the Audio element
	// volume control, has to be lower than 0 dB FS to avoid values
	// outside -1 - 1.
	const generatedProbeDBFS = maxLevel - fullScaleSineLevel;

	// Some elements that we reuse multiple times.
	const probeAudio = document.getElementById("probe-audio");
	const combinedAudio = document.getElementById("combined-audio");
	const raiseProbe = document.getElementById("raise-probe");
	const lowerProbe = document.getElementById("lower-probe");
	const playProbe = document.getElementById("play-probe");
	const playCombined = document.getElementById("play-combined");
	const calibrationAudio = document.getElementById("calibration-audio");
	const erbWidthInput = document.getElementById("erb-width");
	const maskLevelInput = document.getElementById("mask-level");
	const probeLevelInput = document.getElementById("probe-level");
	const probeFrequencyInput = document.getElementById("probe-frequency");
	const erbApartInput = document.getElementById("erb-apart");
	const sineTypeInput = document.getElementById("sine-type");
	const whiteNoiseTypeInput = document.getElementById("white-noise-type");
	const exactMaskFrequenciesInput = document.getElementById(
		"exact-mask-frequencies"
	);

	configureInput(erbWidthInput, runtimeArguments.ERBWidth, 0.5);
	configureInput(maskLevelInput, runtimeArguments.MaskLevels, 80.0);
	configureInput(probeLevelInput, runtimeArguments.ProbeLevel, 60.0);
	configureInput(
		probeFrequencyInput,
		runtimeArguments.ProbeFrequency,
		1000.0
	);
	configureInput(erbApartInput, runtimeArguments.ERBApart, 1.0);
	configureInput(
		exactMaskFrequenciesInput,
		runtimeArguments.ExactMaskFrequencies,
		""
	);
	sineTypeInput.checked = false;
	whiteNoiseTypeInput.checked = true;
	if (runtimeArguments.SignalType) {
		if (runtimeArguments.SignalType == "sine") {
			sineTypeInput.checked = true;
			whiteNoiseTypeInput.checked = false;
		} else if (runtimeArguments.SignalType == "white-noise") {
			sineTypeInput.checked = false;
			whiteNoiseTypeInput.checked = true;
		}

		sineTypeInput.disabled = true;
		whiteNoiseTypeInput.disabled = true;
	}
	if (runtimeArguments.HideControls) {
		document.getElementById("controls").style.display = "none";
	}

	const signalType = (_) => {
		if (sineTypeInput.checked) {
			return "sine";
		}
		return "white-noise";
	};
	const encodeSignalSpec = (spec) => {
		return escape(JSON.stringify(spec));
	};
	const createSingleSignalSpec = (signal) => {
		switch (signalType()) {
			case "sine":
				return {
					Type: "Signal",
					Params: {
						Onset: {
							Delay: signal.delay,
							Duration: 0.1,
						},
						Frequency: signal.frequency,
						Level: signal.level - fullScaleSineLevel,
					},
				};
				break;
			case "white-noise":
				const erbWidth = Number.parseFloat(erbWidthInput.value);
				const width = erbWidthAtHz(signal.frequency) * erbWidth;
				const lowLimit = signal.frequency - width * 0.5;
				const highLimit = signal.frequency + width * 0.5;
				return {
					Type: "Noise",
					Params: {
						Onset: {
							Delay: signal.delay,
							Duration: 0.1,
						},
						LowerLimit: lowLimit,
						UpperLimit: highLimit,
						Level: signal.level - fullScaleSineLevel,
					},
				};
				break;
		}
	};
	const createSignalSpec = (signals) => {
		if (signals.length == 1) {
			return createSingleSignalSpec(signals[0]);
		} else {
			return {
				Type: "Superposition",
				Params: signals.map((signal) => {
					switch (signalType()) {
						case "sine":
							return {
								Type: "Signal",
								Params: {
									Onset: {
										Delay: signal.delay,
										Duration: 0.1,
									},
									Frequency: signal.frequency,
									Level: signal.level - fullScaleSineLevel,
								},
							};
							break;
						case "white-noise":
							const erbWidth = Number.parseFloat(
								erbWidthInput.value
							);
							const width =
								erbWidthAtHz(signal.frequency) * erbWidth;
							const lowLimit = signal.frequency - width * 0.5;
							const highLimit = signal.frequency + width * 0.5;
							return {
								Type: "Noise",
								Params: {
									Onset: {
										Delay: signal.delay,
										Duration: 0.1,
									},
									LowerLimit: lowLimit,
									UpperLimit: highLimit,
									Level: signal.level - fullScaleSineLevel,
								},
							};
							break;
					}
				}),
			};
		}
	};
	calibrationAudio.src =
		"/signal/" +
		escape(
			JSON.stringify({
				Type: "Signal",
				Params: {
					Frequency: 1000,
					Level: 0,
				},
			})
		) +
		".calib.wav";

	// The interesting frequency range, limited at the lower end
	// by reasonable human hearing and at the upper end by
	// reasonable headphone performance.
	const minFrequency = 100.0;
	const maxFrequency = 8000.0;

	// Start the probe at a random level.
	let equivalentProbeLevel = 0;
	const randomizeEquivalentProbeLevel = (_) => {
		equivalentProbeLevel = 40 + 20 * Math.random();
	};
	randomizeEquivalentProbeLevel();

	// Convert level to gain.
	const levelToGain = (level) => {
		return Math.pow(10.0, (level - fullScaleSineLevel) / 20.0);
	};
	// Set the probe audio element to represent the wanted equivalentProbeLevel.
	const setProbeLevel = (_) => {
		probeAudio.volume = levelToGain(
			equivalentProbeLevel - generatedProbeDBFS
		);
	};
	setProbeLevel();

	// Raise probe one dB.
	const raiseProbeFunc = (_) => {
		if (equivalentProbeLevel + 1 <= maxLevel) {
			equivalentProbeLevel++;
			setProbeLevel();
			raiseProbe.style.color = "red";
			setTimeout((_) => {
				raiseProbe.style.color = "black";
			}, 100);
		}
	};

	// Lower probe one dB.
	const lowerProbeFunc = (_) => {
		if (equivalentProbeLevel - 1 >= -20) {
			equivalentProbeLevel--;
			setProbeLevel();
			lowerProbe.style.color = "red";
			setTimeout((_) => {
				lowerProbe.style.color = "black";
			}, 100);
		}
	};

	// Pause the probe player.
	const pauseProbeFunc = (_) => {
		probeAudio.pause();
		playProbe.style.color = "black";
	};

	// Toggle probe player between play and pause.
	const playPauseProbeFunc = (_) => {
		if (!active) return;
		if (probeAudio.paused) {
			probeAudio.currentTime = 0.0;
			probeAudio.play();
			combinedAudio.pause();
			playProbe.style.color = "red";
			playCombined.style.color = "black";
		} else {
			pauseProbeFunc();
		}
	};

	// Pause combined player.
	const pauseCombinedFunc = (_) => {
		combinedAudio.pause();
		playCombined.style.color = "black";
	};

	// Toggle combined player between pause and play.
	const playPauseCombinedFunc = (_) => {
		if (!active) return;
		if (combinedAudio.paused) {
			combinedAudio.currentTime = 0.0;
			combinedAudio.play();
			probeAudio.pause();
			playCombined.style.color = "red";
			playProbe.style.color = "black";
		} else {
			pauseCombinedFunc();
		}
	};

	const datapoints = [];
	let currEvaluation = {};
	const unfinishedEvaluations = [];
	let currentEvaluation = null;
	const finishedEvaluations = [];

	const updateUndoButton = (_) => {
		if (finishedEvaluations.length > 0) {
			document.getElementById("undo").disabled = false;
		} else {
			document.getElementById("undo").disabled = true;
		}
	};
	updateUndoButton();

	document.getElementById("undo").addEventListener("click", (_) => {
		fetch(
			new Request("/log", {
				method: "DELETE",
			})
		).then((resp) => {
			unfinishedEvaluations.unshift(currentEvaluation);
			currentEvaluation = finishedEvaluations.pop();
			plotLog();
			updateUndoButton();
			currentEvaluation();
		});
	});

	// Remove one evaluation from the array of evaluations to run, and run it.
	const runNextEvaluation = (_) => {
		if (currentEvaluation) {
			finishedEvaluations.push(currentEvaluation);
		}
		if (unfinishedEvaluations.length > 0) {
			currentEvaluation = unfinishedEvaluations.shift();
			updateUndoButton();
			currentEvaluation();
		} else {
			currentEvaluation = null;
			pauseProbeFunc();
			pauseCombinedFunc();
			alert(
				"All evaluations finished, click restart to get a new set of evaluations."
			);
		}
	};

	// Generate the list of evaluations and run the first one.
	const restart = (_) => {
		datapoints.length = 0;
		currEvaluation = {
			Calibration: {},
			Run: {
				ID:
					"" +
					new Date().getTime() +
					"_" +
					Math.floor(Math.random() * 2 ** 32),
			},
			Evaluation: {
				Probe: createSignalSpec([
					{
						id: "probe",
						delay: 0.0,
						frequency: Number.parseFloat(probeFrequencyInput.value),
						level: fullScaleSineLevel + generatedProbeDBFS,
					},
				]),
			},
			Results: {},
			probeLevel: Number.parseFloat(probeLevelInput.value),
			maskLevel: Number.parseFloat(maskLevelInput.value),
			probeFrequency: Number.parseFloat(probeFrequencyInput.value),
			erbApart: Number.parseFloat(erbApartInput.value),
		};
		probeAudio.src =
			"/signal/" +
			encodeSignalSpec(currEvaluation.Evaluation.Probe) +
			".probe.wav";
		unfinishedEvaluations.length = 0;
		finishedEvaluations.length = 0;
		const addEvaluationForFrequency = (frequency) => {
			unfinishedEvaluations.push((_) => {
				active = true;
				randomizeEquivalentProbeLevel();
				setProbeLevel();
				pauseProbeFunc();
				pauseCombinedFunc();
				currEvaluation.Evaluation.ID =
					"" +
					new Date().getTime() +
					"_" +
					Math.floor(Math.random() * 2 ** 32);
				currEvaluation.Evaluation.Frequency = frequency;
				currEvaluation.Evaluation.Combined = createSignalSpec([
					{
						id: "probe",
						delay: 0.5,
						frequency: currEvaluation.probeFrequency,
						level: currEvaluation.probeLevel,
					},
					{
						id: "mask",
						delay: 0.0,
						frequency: frequency,
						level: currEvaluation.maskLevel,
					},
				]);
				combinedAudio.src =
					"/signal/" +
					encodeSignalSpec(currEvaluation.Evaluation.Combined) +
					".combined.wav";
				document.getElementById("currently").innerText =
					"Masker at " + frequency + "Hz";
			});
		};
		if (exactMaskFrequenciesInput.value) {
			exactMaskFrequenciesInput.value
				.split(",")
				.map(Number.parseFloat)
				.forEach(addEvaluationForFrequency);
		} else {
			let erb = 4; // 123.08Hz
			for (
				let frequency = erbToHz(erb);
				frequency < maxFrequency;
				frequency = erbToHz(erb)
			) {
				addEvaluationForFrequency(frequency);
				erb += currEvaluation.erbApart;
			}
		}
		runNextEvaluation();
	};

	document.getElementById("restart").addEventListener("click", restart);

	// Plot the current log on local disk.
	const plotLog = (_) => {
		fetch(
			new Request("/log", {
				method: "GET",
			})
		).then((resp) => {
			resp.text().then((body) => {
				const runs = {};
				body.split("\n").forEach((line) => {
					if (line) {
						const measurement = JSON.parse(line);
						if (
							measurement.EntryType ==
							"EquivalentLoudnessMeasurement"
						) {
							if (!runs[measurement.Run.ID]) {
								runs[measurement.Run.ID] = [measurement];
							} else {
								runs[measurement.Run.ID].push(measurement);
							}
						}
					}
				});
				const plots = Object.keys(runs).map((runID) => {
					const x = [];
					const y = [];
					runs[runID].forEach((evaluation) => {
						y.push(
							evaluation.Results.ProbeDBSPLForEquivalentLoudness
						);
						x.push(evaluation.Evaluation.Frequency);
					});
					return {
						x: x,
						y: y,
						type: "scattergl",
						mode: "line",
					};
				});
				Plotly.react("plot", plots, {
					xaxis: {
						title: "Hz",
					},
					yaxis: {
						title: "dB",
					},
				});
			});
		});
	};
	// Record the equivalent loudness of the probe in the two players.
	const recordEquivalentLoudness = (_) => {
		if (!active) return;
		active = false;
		currEvaluation.Results.ProbeGainForEquivalentLoudness =
			probeAudio.volume;
		currEvaluation.Results.ProbeDBSPLForEquivalentLoudness = equivalentProbeLevel;
		currEvaluation.Calibration.FullScaleSineDBSPL = fullScaleSineLevel;
		fetch(
			new Request("/log", {
				method: "POST",
				body: JSON.stringify(currEvaluation),
			})
		).then((resp) => {
			if (resp.status != 200) {
				alert("Unable to record evaluation, see JS console.");
				return;
			}
			plotLog();
			runNextEvaluation();
		});
	};

	document
		.getElementById("equivalent-loudness")
		.addEventListener("click", recordEquivalentLoudness);

	raiseProbe.addEventListener("click", raiseProbeFunc);
	lowerProbe.addEventListener("click", lowerProbeFunc);
	playProbe.addEventListener("click", playPauseProbeFunc);
	playCombined.addEventListener("click", playPauseCombinedFunc);

	[90, 75, 60].forEach((calibrationLevel) => {
		const button = document.getElementById(
			"play-calibration-" + calibrationLevel
		);
		button.style.color = "black";
		const audio = document.getElementById("calibration-audio");
		button.addEventListener("click", (ev) => {
			if (button.style.color == "black") {
				audio.currentTime = 0.0;
				audio.volume = levelToGain(calibrationLevel);
				audio.play();
				button.style.color = "red";
				[90, 75, 60].forEach((otherLevel) => {
					if (otherLevel != calibrationLevel) {
						document.getElementById(
							"play-calibration-" + otherLevel
						).style.color = "black";
					}
				});
			} else {
				audio.pause();
				button.style.color = "black";
			}
		});
	});

	document.addEventListener("keydown", (ev) => {
		if (ignoreKeyEvents) {
			return;
		}
		switch (ev.key) {
			case "ArrowUp":
				ev.preventDefault();
				ev.stopPropagation();
				raiseProbeFunc();
				break;
			case "ArrowDown":
				ev.preventDefault();
				ev.stopPropagation();
				lowerProbeFunc();
				break;
			case "ArrowLeft":
				ev.preventDefault();
				ev.stopPropagation();
				playPauseProbeFunc();
				break;
			case "ArrowRight":
				ev.preventDefault();
				ev.stopPropagation();
				playPauseCombinedFunc();
				break;
			case "z":
				ev.preventDefault();
				ev.stopPropagation();
				recordEquivalentLoudness();
				break;
		}
	});

	plotLog();
	restart();
}
