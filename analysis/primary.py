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

def zscore(series):
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd

def lr_test(full_model, reduced_model, df=1):
    statistic = 2 * (full_model.log_likelihood_ - reduced_model.log_likelihood_)
    p_value = stats.chi2.sf(max(statistic, 0), df)
    return float(statistic), float(p_value)
core = pd.read_excel(CORE_FILE, sheet_name=CORE_SHEET)
core["mean_alps_calc"] = (core.alps_L + core.alps_R) / 2
core["absolute_ai_raw"] = (core.alps_L - core.alps_R).abs()
core["log_wmh"] = h_wmh(core.WMH)
for name in ["AGE_INDEX", "EDUCATION", "BASE_MMSE", "BRAIN_PARENCHYMA_FRACTION", "log_wmh", "mean_alps_calc", "absolute_ai_raw"]:
    core[name+"_z"] = zscore(core[name])

cox_cols = [
    "TIME_BY_DAYS", "EVENT_ADconversion", "BASE_DX", "AGE_INDEX_z", "SEX(M = 0)",
    "EDUCATION_z", "BASE_MMSE_z", "log_wmh_z", "BRAIN_PARENCHYMA_FRACTION_z",
    "mean_alps_calc_z", "absolute_ai_raw_z"
]
cox_df = core[cox_cols].dropna().copy()
cox_df = cox_df.rename(columns={"SEX(M = 0)": "sex_code"})

base_terms = ["AGE_INDEX_z", "sex_code", "EDUCATION_z", "BASE_MMSE_z", "log_wmh_z", "BRAIN_PARENCHYMA_FRACTION_z", "mean_alps_calc_z"]
full_terms = base_terms + ["absolute_ai_raw_z"]

cox_base = CoxPHFitter()
cox_base.fit(cox_df, duration_col="TIME_BY_DAYS", event_col="EVENT_ADconversion", strata=["BASE_DX"], formula=" + ".join(base_terms), robust=True)
cox_full = CoxPHFitter()
cox_full.fit(cox_df, duration_col="TIME_BY_DAYS", event_col="EVENT_ADconversion", strata=["BASE_DX"], formula=" + ".join(full_terms), robust=True)

lr_chi2, lr_p = lr_test(cox_full, cox_base, df=1)
cox_summary = cox_full.summary.reset_index().rename(columns={"covariate": "term"})
cox_summary["n"] = len(cox_df)
cox_summary["events"] = int(cox_df["EVENT_ADconversion"].sum())
cox_summary["absolute_AI_LR_chi2"] = lr_chi2
cox_summary["absolute_AI_LR_p"] = lr_p
display(cox_summary)
save_table(cox_summary, "04_primary_Cox_full_cohort.csv")

ph = proportional_hazard_test(cox_full, cox_df, time_transform="rank")
ph_table = ph.summary.reset_index().rename(columns={"index": "term"})
display(ph_table)
save_table(ph_table, "05_primary_Cox_PH_test.csv")

