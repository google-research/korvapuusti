"use strict";

/*
 * Loudness convers from dB SPL to ISO226 loudness and back.
 */
class Loudness {
	// The frequencies the ISO226 table has values for.
	static _loudX = [
		20,
		25,
		31.5,
		40,
		50,
		63,
		80,
		100,
		125,
		160,
		200,
		250,
		315,
		400,
		500,
		630,
		800,
		1000,
		1250,
		1600,
		2000,
		2500,
		3150,
		4000,
		5000,
		6300,
		8000,
		10000,
		12500,
	];

	// These constants are calculated using scipy interpolation of the ISO226 table values.
	static _af = [
		0.532,
		0.506,
		0.48,
		0.455,
		0.432,
		0.409,
		0.387,
		0.367,
		0.349,
		0.33,
		0.315,
		0.301,
		0.288,
		0.276,
		0.267,
		0.259,
		0.253,
		0.25,
		0.246,
		0.244,
		0.243,
		0.243,
		0.243,
		0.242,
		0.242,
		0.245,
		0.254,
		0.271,
		0.301,
	];

	static _lu = [
		-31.6,
		-27.2,
		-23.0,
		-19.1,
		-15.9,
		-13.0,
		-10.3,
		-8.1,
		-6.2,
		-4.5,
		-3.1,
		-2.0,
		-1.1,
		-0.4,
		0.0,
		0.3,
		0.5,
		0.0,
		-2.7,
		-4.1,
		-1.0,
		1.7,
		2.5,
		1.2,
		-2.1,
		-7.1,
		-11.2,
		-10.7,
		-3.1,
	];

	static _tf = [
		78.5,
		68.7,
		59.5,
		51.1,
		44.0,
		37.5,
		31.5,
		26.5,
		22.1,
		17.9,
		14.4,
		11.4,
		8.6,
		6.2,
		4.4,
		3.0,
		2.2,
		2.4,
		3.5,
		1.7,
		-1.3,
		-4.2,
		-6.0,
		-5.4,
		-1.5,
		6.0,
		12.6,
		13.9,
		12.3,
	];

	static _loudC = [
		[
			[-1.64819506e-6, 4.57648236e-4, -1.05961622e-3],
			[-1.64819506e-6, 4.57648236e-4, -1.05961622e-3],
			[-1.81339013e-6, 1.86256331e-4, -5.79805927e-4],
			[-1.59600013e-7, 1.05574947e-4, -1.34004391e-4],
			[-1.72548504e-7, 5.78361204e-5, -1.26809992e-4],
			[-2.13466493e-7, 1.12543319e-5, -3.59581919e-5],
			[2.28563467e-8, 1.09858601e-5, -1.8388627e-5],
			[-7.06085723e-8, 2.40194701e-6, -8.33901763e-6],
			[5.21405148e-9, 3.86915459e-6, -6.11827507e-6],
			[-1.56861922e-8, -6.04669022e-7, 2.07986288e-7],
			[4.09007258e-11, 8.00019007e-7, -1.60811033e-6],
			[-2.31762629e-9, 1.06227952e-7, -1.38272538e-8],
			[-5.96826081e-11, 4.74585521e-8, -2.37824883e-7],
			[-7.44328626e-10, 6.58156154e-8, -6.15443918e-8],
			[2.03584482e-12, 4.85365377e-9, -2.91107967e-8],
			[-4.60096415e-11, -5.57743044e-9, -1.01401122e-8],
			[-1.60098414e-10, -4.22497071e-8, 2.33703009e-9],
			[7.95072874e-11, 6.44714204e-8, -5.77119835e-8],
			[-3.28639746e-11, 4.16267216e-9, 2.73733027e-8],
			[6.04926892e-12, -2.44849846e-8, 4.27924794e-10],
			[-3.72002545e-12, 3.05262385e-9, 7.72666727e-10],
			[-3.13580583e-13, 7.38200316e-10, -2.88282052e-10],
			[8.9758675e-13, 3.34936403e-10, -2.31783141e-10],
			[4.8181214e-15, 2.09356428e-10, -2.22200675e-10],
			[9.68351764e-15, 2.2446254e-10, -5.08226199e-10],
			[-3.64978485e-14, 4.2080833e-11, -2.35782063e-11],
			[-1.11035308e-14, -2.44078959e-11, 1.11712084e-10],
			[-1.11035308e-14, -2.44078959e-11, 1.11712084e-10],
		],

		[
			[1.31543045e-4, -2.78856441e-2, 6.48415271e-2],
			[1.06820119e-4, -2.10209205e-2, 4.89472839e-2],
			[7.4680315e-5, -1.20967799e-2, 2.82847676e-2],
			[2.84388668e-5, -7.34724347e-3, 1.34997165e-2],
			[2.36508664e-5, -4.17999507e-3, 9.47958476e-3],
			[1.69214747e-5, -1.92438637e-3, 4.53399508e-3],
			[6.03468356e-6, -1.35041545e-3, 2.70012729e-3],
			[7.40606436e-6, -6.91263841e-4, 1.59680967e-3],
			[2.11042144e-6, -5.11117815e-4, 9.71383349e-4],
			[2.65789684e-6, -1.04856584e-4, 3.28964466e-4],
			[7.75553786e-7, -1.77416866e-4, 3.53922821e-4],
			[7.81688895e-7, -5.74140151e-5, 1.12706272e-4],
			[3.29751768e-7, -3.66995645e-5, 1.10009957e-4],
			[3.14532703e-7, -2.45976337e-5, 4.93646122e-5],
			[9.12341149e-8, -4.85294908e-6, 3.09012947e-5],
			[9.20280944e-8, -2.9600241e-6, 1.9548084e-5],
			[6.85631772e-8, -5.80451363e-6, 1.43766268e-5],
			[-2.74958712e-8, -3.11543379e-5, 1.57788448e-5],
			[3.21345944e-8, 1.71992274e-5, -2.75051428e-5],
			[-2.37257893e-9, 2.15700332e-5, 1.23682501e-6],
			[4.88654376e-9, -7.81194833e-6, 1.75033476e-6],
			[-6.93494405e-10, -3.23301256e-6, 2.90933485e-6],
			[-1.30497654e-9, -1.79352194e-6, 2.34718485e-6],
			[9.83869672e-10, -9.39434115e-7, 1.75613784e-6],
			[9.98324037e-10, -3.11364831e-7, 1.08953581e-6],
			[1.03608976e-9, 5.64039075e-7, -8.92546363e-7],
			[8.49950728e-10, 7.78651324e-7, -1.01279521e-6],
			[7.83329543e-10, 6.32203948e-7, -3.42522709e-7],
		],

		[
			[-5.81651035e-3, 1.00798701, -2.25771723],
			[-4.62469453e-3, 7.63454192e-1, -1.68877318],
			[-3.44494171e-3, 5.48189139e-1, -1.18676484],
			[-2.56842867e-3, 3.8291494e-1, -8.31596726e-1],
			[-2.04753133e-3, 2.67642555e-1, -6.01803713e-1],
			[-1.5200909e-3, 1.88285596e-1, -4.19627175e-1],
			[-1.12983621e-3, 1.32613965e-1, -2.96647095e-1],
			[-8.61021251e-4, 9.17803791e-2, -2.10708356e-1],
			[-6.23109106e-4, 6.17208377e-2, -1.4650353e-1],
			[-4.56217966e-4, 4.01617338e-2, -1.00991357e-1],
			[-3.18879941e-4, 2.88707958e-2, -7.36758652e-2],
			[-2.41017807e-4, 1.71292517e-2, -5.03444106e-2],
			[-1.68774164e-4, 1.10118691e-2, -3.58678557e-2],
			[-1.14009984e-4, 5.80160722e-3, -2.23210173e-2],
			[-7.34333023e-5, 2.85654894e-3, -1.42944266e-2],
			[-4.9609215e-5, 1.84086243e-3, -7.73600739e-3],
			[-2.23086989e-5, 3.50891011e-4, -1.96880656e-3],
			[-1.40952377e-5, -7.0408793e-3, 4.06228776e-3],
			[-1.29355569e-5, -1.05296569e-2, 1.13071327e-3],
			[-2.51885145e-6, 3.03958427e-3, -8.06319797e-3],
			[-1.51326552e-6, 8.5428182e-3, -6.86833406e-3],
			[5.83259159e-7, 3.02033776e-3, -4.53849926e-3],
			[-7.15746955e-7, -2.46909665e-4, -1.12176145e-3],
			[-9.88687794e-7, -2.56992231e-3, 2.36606284e-3],
			[9.93505915e-7, -3.82072126e-3, 5.21173649e-3],
			[3.63824384e-6, -3.49224474e-3, 5.46782277e-3],
			[6.84451267e-6, -1.20967106e-3, 2.22874209e-3],
			[1.01110732e-5, 1.61203948e-3, -4.81893755e-4],
		],

		[
			[5.32e-1, -3.16e1, 7.85e1],
			[5.06e-1, -2.72e1, 6.87e1],
			[4.8e-1, -2.3e1, 5.95e1],
			[4.55e-1, -1.91e1, 5.11e1],
			[4.32e-1, -1.59e1, 4.4e1],
			[4.09e-1, -1.3e1, 3.75e1],
			[3.87e-1, -1.03e1, 3.15e1],
			[3.67e-1, -8.1, 2.65e1],
			[3.49e-1, -6.2, 2.21e1],
			[3.3e-1, -4.5, 1.79e1],
			[3.15e-1, -3.1, 1.44e1],
			[3.01e-1, -2.0, 1.14e1],
			[2.88e-1, -1.1, 8.6],
			[2.76e-1, -4.0e-1, 6.2],
			[2.67e-1, 0.0, 4.4],
			[2.59e-1, 3.0e-1, 3.0],
			[2.53e-1, 5.0e-1, 2.2],
			[2.5e-1, 0.0, 2.4],
			[2.46e-1, -2.7, 3.5],
			[2.44e-1, -4.1, 1.7],
			[2.43e-1, -1.0, -1.3],
			[2.43e-1, 1.7, -4.2],
			[2.43e-1, 2.5, -6.0],
			[2.42e-1, 1.2, -5.4],
			[2.42e-1, -2.1, -1.5],
			[2.45e-1, -7.1, 6.0],
			[2.54e-1, -1.12e1, 1.26e1],
			[2.71e-1, -1.07e1, 1.39e1],
		],
	];

	// Calculates the correct constants for a particular frequency.
	static _loudInterp(f) {
		let interval = 0;
		while (
			interval < Loudness._loudX.length - 1 &&
			f >= Loudness._loudX[interval + 1]
		) {
			interval++;
		}
		if (interval == 0 || interval == Loudness._loudX.length - 1) {
			return [
				Loudness._af[interval],
				Loudness._lu[interval],
				Loudness._tf[interval],
			];
		}
		let af = 0;
		let lu = 0;
		let tf = 0;
		for (let order = 0; order < Loudness._loudC.length; order++) {
			if (interval > Loudness._loudC[order].length - 1) {
				interval = Loudness._loudC[order].length - 1;
			}
			let iC = Loudness._loudC[order][interval];
			let fD = Math.pow(
				f - Loudness._loudX[interval],
				Loudness._loudC.length - order - 1
			);
			af += iC[0] * fD;
			lu += iC[1] * fD;
			tf += iC[2] * fD;
		}
		return [af, lu, tf];
	}
	static loud2spl(phons, freq) {
		if (freq > 11000) {
			throw "Loudness undefined for frequencies > 11kHz.";
		}
		let c = Loudness._loudInterp(freq);
		let af = 4.47e-3 * (10 ** (0.025 * phons) - 1.15);
		af = af + (0.4 * 10 ** ((c[2] + c[1]) / 10 - 9)) ** c[0];
		return (10 / c[0]) * Math.log10(af) - c[1] + 94;
	}
	static spl2loud(spl, freq) {
		if (freq > 11000) {
			throw "Loudness undefined for frequencies > 11kHz.";
		}
		let c = Loudness._loudInterp(freq);
		let expf = function (x) {
			return (0.4 * 10 ** ((x + c[1]) / 10 - 9)) ** c[0];
		};
		let bf = expf(spl) - expf(c[2]) + 0.005135;
		return 40 * Math.log10(bf) + 94;
	}
}

// See https://en.wikipedia.org/wiki/Equivalent_rectangular_bandwidth
function erbWidthAtHz(f) {
	return 24.7 * (4.37 * f * 0.001 + 1);
}
function hzToERB(f) {
	return 21.4 * Math.log10(1 + 0.00437 * f);
}
// Simply an inversion of hzToERB.
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
	const extraMasksInput = document.getElementById("extra-masks");
	const xAxisLinearHzInput = document.getElementById("x-axis-linear-hz");
	const xAxisLogHzInput = document.getElementById("x-axis-log-hz");
	const xAxisCamsInput = document.getElementById("x-axis-cams");
	const yAxisDBSPLInput = document.getElementById("y-axis-db-spl");
	const yAxisPhonsInput = document.getElementById("y-axis-phons");

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
	configureInput(extraMasksInput, runtimeArguments.ExtraMasks, "");
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
				const combinedSounds = [
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
				];
				extraMasksInput.value.split(",").forEach((extraMaskSpec) => {
					if (extraMaskSpec != "") {
						const splitExtraMask = extraMaskSpec.split("/");
						combinedSounds.push({
							id: "mask",
							delay: 0.0,
							frequency: Number.parseFloat(splitExtraMask[0]),
							level: Number.parseFloat(splitExtraMask[1]),
						});
					}
				});
				currEvaluation.Evaluation.Combined = createSignalSpec(
					combinedSounds
				);
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
						if (yAxisDBSPLInput.checked) {
							y.push(
								evaluation.Results
									.ProbeDBSPLForEquivalentLoudness
							);
						} else {
							y.push(
								Loudness.spl2loud(
									evaluation.Results
										.ProbeDBSPLForEquivalentLoudness,
									evaluation.Evaluation.Frequency
								)
							);
						}
						if (xAxisCamsInput.checked) {
							x.push(hzToERB(evaluation.Evaluation.Frequency));
						} else {
							x.push(evaluation.Evaluation.Frequency);
						}
					});
					return {
						x: x,
						y: y,
						type: "scattergl",
						mode: "line",
					};
				});
				const xAxis = {};
				if (xAxisLinearHzInput.checked) {
					xAxis.title = "Hz";
				} else if (xAxisLogHzInput.checked) {
					xAxis.title = "Hz";
					xAxis.type = "log";
				} else if (xAxisCamsInput.checked) {
					xAxis.title = "Cams";
				}
				const yAxis = {};
				if (yAxisDBSPLInput.checked) {
					yAxis.title = "dB SPL";
				} else if (yAxisPhonsInput.checked) {
					yAxis.title = "Phons";
				}
				Plotly.react("plot", plots, {
					xaxis: xAxis,
					yaxis: yAxis,
				});
			});
		});
	};

	[
		xAxisLinearHzInput,
		xAxisLogHzInput,
		xAxisCamsInput,
		yAxisDBSPLInput,
		yAxisPhonsInput,
	].forEach((inp) => {
		inp.addEventListener("change", plotLog);
	});

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
