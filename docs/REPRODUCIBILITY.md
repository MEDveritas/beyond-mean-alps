# Reproducibility

## Inputs and outputs

The analyses start from the final derived datasets specified in the [data dictionary](DATA_DICTIONARY.md). Inputs contain one baseline record per participant, derived imaging measurements, and the follow-up outcomes required by each module. Participant-level data and checkpoints must be supplied separately.

The runner records input hashes, software versions, analysis settings, and section completion in `run_manifest.json`. Each section writes an execution log. The manuscript export generates table files and statistical values for figures. Figure 1/Table S1 counts are reported cohort-selection aggregates. Raw MRI processing and final composite figure assembly are outside this workflow.

## Training and saved predictions

The default run fits the statistical models and performs 450 neural-model fits. Completed fold checkpoints can be reused when the input data, configuration, and source signature match. Use separate output directories for runs with different settings or code.

To evaluate a complete saved OOF prediction run, provide its directory:

```bash
python run_analysis.py --data-dir data --output-dir results --prediction-cache /path/to/saved_prediction_run
python export_manuscript.py --data-dir data --results results
python validate_results.py --results results
```

The saved run must contain `training_manifest.json` and a `checkpoints/` directory with `master_splits.csv` and 150 fold prediction files, each with its matching JSON metadata. Re-evaluation checks the data hash, model configuration, feature sets, fold membership, inner-training and validation participants, scaling, and censoring weights. It reads the supplied checkpoints without modifying them and requires a complete set of folds.

The prediction configuration includes an internal compatibility identifier used to validate checkpoints. Changing it or other training settings requires compatible checkpoints or a fresh run.

## Numerical comparison

The `reference/` directory contains 26 aggregate result tables and a manifest defining the result paths, row keys, numeric columns, and reference-file checksums. Comparison uses an absolute tolerance of `1e-8` and relative tolerance of `1e-7`.

```bash
python validate_results.py --results results --skip-prediction
```

Use this command to compare association and sensitivity outputs separately after fresh neural training. Prediction references correspond to the saved study OOF predictions. Neural training can vary with numerical libraries, thread scheduling, and hardware, so a fresh training run may not match these references exactly.

The paired prediction bootstrap holds OOF predictions and censoring weights fixed. Its confidence intervals do not include uncertainty from refitting the models or censoring distributions. See [Analysis methods](ANALYSIS_METHODS.md) for evaluation details.

## Tests

```bash
python -m unittest discover -s tests -v
```

These tests cover preprocessing, censoring weights, AUC ties, Brier normalization, paired resampling, multiplicity correction, checkpoint validation, Python syntax, and notebook/reference structure. They run without participant data. Reproducing the fitted study results requires the private analysis inputs.

## Output handling

Use `results/` or another private output directory. Generated files can contain participant identifiers and individual predictions. The default input and result folders are excluded from Git; custom output directories require their own exclusion rules. Keep independent runs in separate directories when their outputs need to be preserved.
