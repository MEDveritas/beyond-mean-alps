from __future__ import annotations
from pathlib import Path
import json, warnings, re
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# DATA_DIR, SECTION_OUT and SETTINGS are supplied by run_analysis.py.
OUT = OUTPUT_DIR = TABLE_DIR = SECTION_OUT
OUT.mkdir(parents=True, exist_ok=True)
CORE_FILE = COHORT_PATH = DATA_DIR / "baseline_clinical_alps.xlsx"
CORE_SHEET = SHEET = SHEET_NAME = "survival_dataset_final_age"

def display(*values):
    """Tables are written to CSV; console previews are disabled."""
    pass

def h_wmh(x):
    x = pd.to_numeric(x, errors="coerce")
    if (x < 0).any():
        raise ValueError("Negative WMH volume")
    return np.log1p(x)

def save_table(frame, filename, index=False):
    frame.to_csv(OUT / filename, index=index)

from dataclasses import dataclass
from typing import Sequence
from scipy.stats import chi2
import platform
EXPECTED_LIFELINES_VERSION="0.30.0"
SPLINE_KNOT_QUANTILES=(.1,.5,.9)
SPLINE_GRID_QUANTILES=(.01,.99)
raw=pd.read_excel(DATA_DIR/"reoriented_clinical_alps.xlsx",sheet_name=SHEET)
cohort=raw.rename(columns={"TIME_BY_DAYS":"duration_days","EVENT_ADconversion":"event","BASE_DX":"baseline_dx","AGE_INDEX":"age","SEX(M = 0)":"sex","EDUCATION":"education","BASE_MMSE":"baseline_mmse","BRAIN_PARENCHYMA_FRACTION":"bpf","ALPS":"mean_alps"}).copy()
cohort["log_wmh"]=h_wmh(cohort.WMH)
cohort["abs_asymmetry"]=(cohort.alps_L-cohort.alps_R).abs()

def restricted_cubic_spline_basis(x: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    """Harrell-style restricted cubic spline basis: linear x + K-2 nonlinear terms."""
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    if len(k) < 3 or np.any(np.diff(k) <= 0):
        raise ValueError("Need at least 3 strictly increasing knots")
    scale = (k[-1] - k[0]) ** 2
    if scale <= 0:
        raise ValueError("Degenerate spline knots")
    cols = [x]
    tp = lambda value, knot: np.maximum(value - knot, 0.0) ** 3
    for j in range(len(k) - 2):
        h = (
            tp(x, k[j])
            - (k[-1] - k[j]) / (k[-1] - k[-2]) * tp(x, k[-2])
            + (k[-2] - k[j]) / (k[-1] - k[-2]) * tp(x, k[-1])
        ) / scale
        cols.append(h)
    return np.column_stack(cols)


@dataclass(frozen=True)
class SplineResult:
    curve: pd.DataFrame
    metadata: pd.DataFrame
    coefficients: pd.DataFrame


def fit_adjusted_spline(cohort_df: pd.DataFrame) -> SplineResult:
    try:
        import lifelines
        from lifelines import CoxPHFitter
    except ImportError as exc:
        raise ImportError(
            "Install the required dependency: "
            "pip install lifelines==0.30.0"
        ) from exc
    if lifelines.__version__ != EXPECTED_LIFELINES_VERSION:
        print(
            f"WARNING: lifelines {lifelines.__version__} is installed; this analysis requires "
            f"{EXPECTED_LIFELINES_VERSION}. Install the required version before fitting."
        )

    required = [
        "duration_days", "event", "baseline_dx", "age", "sex", "education",
        "baseline_mmse", "log_wmh", "bpf", "mean_alps", "abs_asymmetry"
    ]
    d = cohort_df[required].dropna().copy()
    if d.empty or d["event"].sum() == 0:
        raise ValueError("No complete event-bearing observations for the spline model")

    continuous_covariates = ["age", "education", "baseline_mmse", "log_wmh", "bpf", "mean_alps"]
    scaling_rows = []
    for col in continuous_covariates + ["abs_asymmetry"]:
        mean = float(d[col].mean())
        sd = float(d[col].std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Cannot standardize {col}: SD={sd}")
        d[f"z_{col}"] = (d[col] - mean) / sd
        scaling_rows.append({"variable": col, "mean": mean, "sd": sd})

    x = d["z_abs_asymmetry"].to_numpy()
    knots = np.quantile(x, SPLINE_KNOT_QUANTILES)
    basis = restricted_cubic_spline_basis(x, knots)
    spline_cols = ["z_abs_asymmetry"] + [f"rcs_abs_asym_nl{i}" for i in range(1, basis.shape[1])]
    for i, col in enumerate(spline_cols):
        d[col] = basis[:, i]

    adjustment_cols = ["sex"] + [f"z_{c}" for c in continuous_covariates]
    common = ["duration_days", "event", "baseline_dx"] + adjustment_cols

    reduced = CoxPHFitter()
    linear = CoxPHFitter()
    spline = CoxPHFitter()
    fit_kwargs = dict(
        duration_col="duration_days", event_col="event",
        strata=["baseline_dx"], robust=True, show_progress=False
    )
    reduced.fit(d[common], **fit_kwargs)
    linear.fit(d[common + ["z_abs_asymmetry"]], **fit_kwargs)
    spline.fit(d[common + spline_cols], **fit_kwargs)

    lr_overall = 2.0 * (spline.log_likelihood_ - reduced.log_likelihood_)
    lr_nonlinear = 2.0 * (spline.log_likelihood_ - linear.log_likelihood_)
    df_overall = len(spline_cols)
    df_nonlinear = len(spline_cols) - 1
    p_overall = float(chi2.sf(max(lr_overall, 0), df_overall))
    p_nonlinear = float(chi2.sf(max(lr_nonlinear, 0), df_nonlinear))

    raw_mean = float(d["abs_asymmetry"].mean())
    raw_sd = float(d["abs_asymmetry"].std(ddof=1))
    raw_lo, raw_hi = np.quantile(d["abs_asymmetry"], SPLINE_GRID_QUANTILES)
    raw_grid = np.linspace(raw_lo, raw_hi, 240)
    z_grid = (raw_grid - raw_mean) / raw_sd
    z_ref = (float(d["abs_asymmetry"].median()) - raw_mean) / raw_sd
    b_grid = restricted_cubic_spline_basis(z_grid, knots)
    b_ref = restricted_cubic_spline_basis(np.array([z_ref]), knots)[0]
    delta_design = b_grid - b_ref

    beta = spline.params_.loc[spline_cols].to_numpy()
    covariance = spline.variance_matrix_.loc[spline_cols, spline_cols].to_numpy()
    log_hr = delta_design @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", delta_design, covariance, delta_design))

    counts, edges = np.histogram(d["abs_asymmetry"], bins=30)
    bin_id = np.clip(np.digitize(raw_grid, edges[1:-1]), 0, len(counts) - 1)
    curve = pd.DataFrame({
        "abs_asymmetry_raw": raw_grid,
        "abs_asymmetry_sd": z_grid,
        "hazard_ratio": np.exp(log_hr),
        "ci_lower": np.exp(log_hr - 1.96 * se),
        "ci_upper": np.exp(log_hr + 1.96 * se),
        "distribution_count": counts[bin_id],
    })

    metadata = pd.DataFrame([
        {
            "n": len(d), "events": int(d["event"].sum()),
            "reference_raw_median": float(d["abs_asymmetry"].median()),
            "raw_mean": raw_mean, "raw_sd": raw_sd,
            "knot_quantiles": "|".join(map(str, SPLINE_KNOT_QUANTILES)),
            "knot_raw_values": "|".join(f"{raw_mean + raw_sd * v:.8g}" for v in knots),
            "lr_overall": lr_overall, "df_overall": df_overall, "p_overall": p_overall,
            "lr_nonlinear": lr_nonlinear, "df_nonlinear": df_nonlinear,
            "p_nonlinear": p_nonlinear,
            "python_version": platform.python_version(),
            "lifelines_version": lifelines.__version__,
        }
    ])
    coefficients = spline.summary.reset_index().rename(columns={"covariate": "term"})
    scaling = pd.DataFrame(scaling_rows)
    scaling.to_csv(OUTPUT_DIR / "figure2b_scaling.csv", index=False)
    return SplineResult(curve=curve, metadata=metadata, coefficients=coefficients)


if cohort is not None:
    spline_result = fit_adjusted_spline(cohort)
    spline_result.curve.to_csv(OUTPUT_DIR / "figure2b_spline_curve.csv", index=False)
    spline_result.metadata.to_csv(OUTPUT_DIR / "figure2b_spline_metadata.csv", index=False)
    spline_result.coefficients.to_csv(OUTPUT_DIR / "figure2b_spline_coefficients.csv", index=False)
    display(spline_result.metadata)
else:
    spline_result = None
