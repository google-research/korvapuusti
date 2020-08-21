# Data Processing

## Krippendorff's Alpha Coefficient

To calculate the Krippendorff's alpha for a particular dataset labeled by more
than one listener, make sure the data is in a json file with the following
format:

```
[
{
        "probe_frequency": 842.0,
        "probe_level": 60,
        "perceived_probe_levels": [
            62,
            58,
            61,
            58
        ],
        "worker_ids": [
            4350346416,
            4350360853,
            4350361963,
            4350362064
        ],
        "masker_frequency": 1370.0,
        "masker_level": 40
    },
]
```

And run:

`>> python3 annotator_agreement.py --input_file_path=<PATH_TO_DATA>`

## Pre-process the Listening Data

The pre-processing of the data means dropping data examples with too high
empirical variance (where too high means the 15% of the answers with the highest
variance) and splitting the data in a training and test set (where the test set
is 15% of the data). To pre-process data, make sure it is in a json file in the
following format:

```
[
{
        "probe_frequency": 842.0,
        "probe_level": 60,
        "perceived_probe_levels": [
            62,
            58,
            61,
            58
        ],
        "worker_ids": [
            4350346416,
            4350360853,
            4350361963,
            4350362064
        ],
        "masker_frequency": 1370.0,
        "masker_level": 40
    },
]
```

And run:

`>> python3 preprocess_data.py --input_file_path=<PATH_TO_DATA>`

To pre-process the listening data created by the open-source listening tool, use
the file `process_ood_sets.py`. This requires specifying the masker frequencies,
masker levels, and probe levels that the data contains, and the data from the
tool should be saved in folders per evaluator (`evaluator_<i>`) named in the
following way:

`mask_<mask frequency>Hz_<mask level>dB_<probe level>dB_signal_shape`

This folder must contain `evaluations.json` with the output of the tool from
this repository (found
[here](https://github.com/google-research/korvapuusti/tree/master/experiments/partial_loudness)).
An example of the directory structure and evaluations file is found in directory
`tool_data` of this folder.
