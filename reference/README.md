# Reference results

The 26 CSV files in this directory contain aggregate study results for comparison with outputs from the analysis modules. Participant-level data and predictions are not included.

`manifest.json` maps each reference file to an analysis output and defines the row keys and numeric columns used by `validate_results.py`. The `reference_sha256` field records the checksum of the packaged reference CSV.

Run the comparison from the repository root after completing the analyses:

```bash
python validate_results.py --results results
```

Prediction reference values correspond to saved study OOF predictions. To compare association and sensitivity results separately after fresh neural training, add `--skip-prediction`. Full instructions are in [Reproducibility](../docs/REPRODUCIBILITY.md).
