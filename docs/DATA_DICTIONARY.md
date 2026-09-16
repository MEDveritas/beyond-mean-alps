# Required private inputs

The repository starts from the study's final derived datasets. It does not download ADNI or reconstruct these files from raw MRI. Keep one baseline row per participant, preserve original precision and row order, and do not redistribute subject-level files with this code.

Input filenames are listed in [data/README.md](../data/README.md). Supply the required files using the schema below.

## 1. Primary workbook

`baseline_clinical_alps.xlsx`, sheet `survival_dataset_final_age`, contains 1,131 unique participants and 115 AD conversions.

| Column | Meaning / coding |
|---|---|
| `PTID` | Unique ADNI participant string used for joins |
| `INDEX_DATE` | Index MRI date; ordinary date or Excel serial date |
| `BASE_DX` | `CN` or `MCI` |
| `AGE_INDEX` | Age at index, years |
| `SEX(M = 0)` | Male 0, female 1 |
| `EDUCATION` | Years of education |
| `BASE_MMSE` | Baseline MMSE score |
| `EVENT_ADconversion` | Conversion event 1; censored 0 |
| `TIME_BY_DAYS` | Positive follow-up duration in days; used by survival models |
| `TIME_BY_MONTHS` | Stored follow-up months, used in Table 1 descriptives; prediction months are instead calculated from days / (365.25/12) |
| `WMH` | Nonnegative WMH volume, mL; transformed with log1p |
| `BRAIN_PARENCHYMA_FRACTION` | Supplied UCD tissue BPF, dimensionless |
| `alps_L`, `alps_R` | Positive left/right ALPS values |
| `ALPS` | Stored bilateral mean, equal to `(alps_L + alps_R)/2` |
| `x_proj_L`, `x_proj_R` | x-direction diffusivity in projection ROI, common native diffusivity units |
| `x_assoc_L`, `x_assoc_R` | x-direction diffusivity in association ROI |
| `y_proj_L`, `y_proj_R` | y-direction diffusivity in projection ROI |
| `z_assoc_L`, `z_assoc_R` | z-direction diffusivity in association ROI |

Core fields must be finite and complete. Asymmetry variables and standardized covariates are derived by the code. Preserve native diffusivity values/units consistently; the component analyses subsequently standardize their raw bilateral absolute differences.

## 2. F2 merged workbook

`reoriented_clinical_alps.xlsx`, same sheet. It contains the same primary fields/participants plus:

- `F2_AVAILABLE`: 1 when the reoriented measurement is available; 1,130 matched participants.
- `F2_alps_L`, `F2_alps_R`, `F2_alps_mean`: reoriented bilateral ALPS and the supplied bilateral mean.
- `F2_FA_proj_L`, `F2_FA_proj_R`, `F2_FA_assoc_L`, `F2_FA_assoc_R`: FA, dimensionless.
- `F2_MD_proj_L`, `F2_MD_proj_R`, `F2_MD_assoc_L`, `F2_MD_assoc_R`: MD, consistent native diffusivity units.
- `SCANNER_MFR`, `THICKNESS`, `TE`, `TR`, `DIRECTIONS`: scanner manufacturer, slice thickness (mm), echo/repetition times and number of diffusion directions used by technical adjustment. Preserve acquisition time units and vendor labels as supplied. Missing `DIRECTIONS` yields the reported common complete-case subset of 936 participants/105 events.

The runner checks equality of primary numerical fields between workbooks when both are needed. Missing F2 scalar measurements are handled within the relevant complete-case analyses.

## 3. FreeSurfer table

`structural_mri_volumes.csv` contains longitudinal FreeSurfer records. Required columns:

| Columns | Use |
|---|---|
| `PTID`, `EXAMDATE` | Participant and scan date matching |
| `STATUS`, `OVERALLQC` | Duplicate priority: complete before partial; Pass, Partial, Hippocampus Only, Fail, missing |
| `ST37SV`, `ST96SV` | Left/right lateral ventricular volume |
| `ST147SV`, `ST148SV` | Left/right cerebral gray-matter volume |
| `ST150SV`, `ST151SV` | Left/right cerebral white-matter volume |
| `ST10CV` | Intracranial volume |
| `ST30SV`, `ST89SV` | Optional inferior lateral ventricular fields; retained in duplicate-completeness ranking when present |

Volume fields share mm³ units. Original ADNI column definitions and the exact exported table must be retained. Structural asymmetry is the absolute bilateral difference divided by bilateral mean. Ventricular burden is normalized to intracranial volume. Matching and missingness produce 1,097 participants/110 events for primary structural Cox models, and 1,101/111 within 365 days.

## 4–6. Technical measurements

All three CSVs require one row per `PTID` and derived values from the study's upstream processing.

| File | Numerical columns |
|---|---|
| `diffusion_motion_metrics.csv` | `mean_relative_rms`, `mean_restricted_relative_rms`, `p95_relative_rms` (displacement), `outlier_slice_fraction` (fraction) |
| `head_orientation_angles.csv` | `pitch_deg`, `yaw_deg`, `roll_deg` (degrees) |
| `acquisition_plane_angles.csv` | `acq_tilt_x_deg`, `acq_tilt_y_deg` (degrees) |

Motion and orientation CSVs can include boolean `motion_success` / `orientation_success`, failure-reason columns, and `duplicate_conflict`. Success flags must be actual booleans or numeric 0/1, not textual `"False"`. Failed values are set to missing. The manuscript run expects complete motion and orientation coverage of all 1,131 participants. The exact upstream displacement convention and image-axis/registration convention must be supplied with the original derived exports; this repository cannot establish them from raw images.

Run `python run_analysis.py --data-dir /private/input --check-inputs` before analysis. Model-specific missing fields produce explicit errors during their section. Input hashes are recorded in the private run manifest; these files are not packaged into the public repository.
