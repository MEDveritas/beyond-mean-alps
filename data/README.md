# Private inputs

Prepare the six datasets below with the indicated filenames and place them here, or pass their directory with `--data-dir`. See [DATA_DICTIONARY.md](../docs/DATA_DICTIONARY.md) for required columns, coding, units, and Excel sheet names.

| Required information | Filename used by the code |
|---|---|
| Baseline clinical, survival and ALPS measurements | `baseline_clinical_alps.xlsx` |
| Baseline data with reoriented ALPS, FA/MD and acquisition metadata | `reoriented_clinical_alps.xlsx` |
| Dated bilateral structural volumes, intracranial volume and QC | `structural_mri_volumes.csv` |
| Within-scan motion and outlier-slice measurements | `diffusion_motion_metrics.csv` |
| Head pitch, yaw and roll | `head_orientation_angles.csv` |
| Acquisition-plane tilt about the x and y axes | `acquisition_plane_angles.csv` |

Input contents must follow the documented schema. Both Excel workbooks use the `survival_dataset_final_age` sheet.

No participant records are distributed in this repository. Files placed here are ignored by Git except this README. Generated outputs belong in `results/`.
