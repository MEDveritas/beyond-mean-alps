# Beyond Mean ALPS

Analysis code for **Greater Absolute DTI-ALPS Asymmetry Is Associated with Higher Risk of Conversion to Alzheimer’s Disease**.

This repository contains the statistical analyses reported in the manuscript, including survival analyses, sensitivity analyses, and deep learning prediction models in ADNI. The cohort comprises **1,131 participants and 115 AD conversions**. The primary exposure is the raw absolute interhemispheric difference, **`abs(ALPS_L - ALPS_R)`**.

한국어 안내: [README.ko.md](README.ko.md).

## Analyses

| Module | Analysis | Manuscript output |
|---|---|---|
| `primary` | Diagnosis-stratified Cox regression and proportional-hazards assessment | Table 2 |
| `directional_outliers` | Hemisphere-specific and tensor-component analyses, tail exclusions, and influence sensitivity | Tables S2–S3; Figure 4A/C |
| `structural` | Structural asymmetry and scan-date matching sensitivity | Table S4; Figure S1D |
| `technical_risk` | Scanner, acquisition, motion, and orientation adjustment; alternative asymmetry definitions; adjusted absolute risks | Tables S7–S8; Figure 3B/C |
| `reorientation_scalar` | Tensor-reorientation agreement and FA/MD asymmetry analyses | Tables S5A–S6; Figure 4B; Figure S1B/C |
| `competition` | Joint modeling of ALPS and association-region Dzz asymmetry | Tables S9A–S9B; Figure 4C |
| `spline` | Restricted cubic spline analysis | Figure 3A |
| `prediction` | Repeated cross-validation of DeepSurv, CoxTime, and DeepHitSingle | Tables S10–S12 |

Model specifications, scaling, and evaluation procedures are described in [Analysis methods](docs/ANALYSIS_METHODS.md).

## Data

The code operates on final derived analysis datasets. Participant-level ADNI data and study-specific MRI measurements must be supplied separately. ADNI access is available through the [ADNI data access process](https://adni.loni.usc.edu/data-samples/adni-data/).

| Input file | Required information |
|---|---|
| `baseline_clinical_alps.xlsx` | Baseline clinical variables, survival outcomes, structural covariates, bilateral ALPS, and directional diffusivities |
| `reoriented_clinical_alps.xlsx` | Baseline data with VECREG-derived ALPS, FA/MD, and acquisition metadata |
| `structural_mri_volumes.csv` | Dated bilateral structural volumes, intracranial volume, and QC fields |
| `diffusion_motion_metrics.csv` | Within-scan displacement and outlier-slice measurements |
| `head_orientation_angles.csv` | Head pitch, yaw, and roll |
| `acquisition_plane_angles.csv` | Acquisition-plane tilt angles |

Place the files in `data/`, or specify their directory with `--data-dir`. Both workbooks use the sheet `survival_dataset_final_age`. Required columns, coding, and units are listed in the [data dictionary](docs/DATA_DICTIONARY.md).

All six inputs are required for a complete run. Individual analysis sections require only their corresponding inputs. The code includes checks for the manuscript cohort; application to another cohort requires adapting those checks and the input mappings.

MRI preprocessing is described in the manuscript and the cited [DTI-ALPS pipeline](https://github.com/gbarisano/alps). This repository starts with derived measurements. Figure 1/Table S1 selection counts are provided as reported aggregates. Exports contain statistical source values for the figures; final composite figure layouts and MRI illustrations are outside the export workflow.

## Installation

Use **Python 3.12** and run commands from the repository root.

```bash
python -m venv .venv
```

Activate the environment using the command for your platform:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` specifies dependency ranges and fixes the lifelines and pycox versions. `requirements-tested.txt` records the analysis environment. CPU is the default device; neural training results can vary across numerical libraries and hardware.

## Run the analyses

Validate the required inputs, run the analyses, and export the tables:

```bash
python run_analysis.py --data-dir data --check-inputs
python run_analysis.py --data-dir data --output-dir results
python export_manuscript.py --data-dir data --results results
```

The complete prediction analysis performs **450 neural-model fits**: 3 model families × 2 predictor sets × 5 repetitions × 5 folds × 3 training seeds. Matching checkpoints allow an interrupted run to resume. Use a new output directory when the source code, configuration, or input data change.

To run selected sections:

```bash
python run_analysis.py --data-dir data --output-dir results --sections primary directional_outliers
```

The manuscript export requires completed outputs from all eight sections.

For VS Code or Jupyter, open [Manuscript_Analyses.ipynb](Manuscript_Analyses.ipynb). It runs the same modules as the command-line interface.

## Results and checks

Analysis outputs are written to section-specific folders under `results/`. The export command writes manuscript-numbered tables and figure source values to `results/manuscript/`. Each section records an execution log, and `run_manifest.json` records inputs, software versions, settings, and completion status.

The [reference results](reference/README.md) contain 26 aggregate CSVs for numerical comparison:

```bash
python validate_results.py --results results --skip-prediction
```

This command compares association and sensitivity results with the references. Prediction reference values correspond to saved OOF predictions; a fresh training run may produce different values. Saved predictions can be evaluated with `--prediction-cache`, as described in [Reproducibility](docs/REPRODUCIBILITY.md).

Run the tests that do not require participant data:

```bash
python -m unittest discover -s tests -v
```

Participant data, individual predictions, and training checkpoints are not distributed. Generated results may contain participant identifiers and should remain in private storage. The default `data/` and `results/` paths are excluded from Git.

## Citation and license

Please cite the associated manuscript when using this code. Citation metadata is provided in [CITATION.cff](CITATION.cff).

The code is available under the [MIT License](LICENSE). ADNI data and third-party software are subject to their respective terms.
