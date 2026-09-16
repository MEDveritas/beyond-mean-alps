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
    p_value = stats.chi2.sf(max(float(statistic), 0.0), df)
    return float(statistic), float(p_value)

def _raw_columns_from_terms(terms):
    """
    Extract raw dataframe columns required by the limited formula syntax used here.
    Supports plain columns and C(column). We deliberately do not guess other expressions.
    """
    cols = []
    for term in terms:
        term = term.strip()
        m = re.fullmatch(r"C\(([^)]+)\)", term)
        if m:
            cols.append(m.group(1).strip())
        else:
            cols.append(term)
    return list(dict.fromkeys(cols))

def fit_stratified_cox(data, terms, label, robust=True, complete_case_terms=None):
    cc_terms = complete_case_terms if complete_case_terms is not None else terms
    needed_raw = _raw_columns_from_terms(cc_terms)
    needed = ["TIME_BY_DAYS", "EVENT_ADconversion", "BASE_DX"] + needed_raw
    missing = [x for x in needed if x not in data.columns]
    if missing:
        raise KeyError(f"{label}: missing columns required by formula: {missing}")

    d = data[needed].dropna().copy()
    cph = CoxPHFitter()
    cph.fit(
        d,
        duration_col="TIME_BY_DAYS",
        event_col="EVENT_ADconversion",
        strata=["BASE_DX"],
        formula=" + ".join(terms),
        robust=robust,
    )
    cph._analysis_label = label
    return cph, d

def fit_nested_cox(data, base_terms, added_terms, label, robust=True):
    """
    Fit reduced and full models on exactly the same complete-case rows.
    This is required for a valid nested likelihood-ratio comparison.
    """
    union_terms = list(dict.fromkeys(base_terms + added_terms))
    needed_raw = _raw_columns_from_terms(union_terms)
    needed = ["TIME_BY_DAYS", "EVENT_ADconversion", "BASE_DX"] + needed_raw
    missing = [x for x in needed if x not in data.columns]
    if missing:
        raise KeyError(f"{label}: missing columns required by nested models: {missing}")

    d = data[needed].dropna().copy()

    reduced = CoxPHFitter()
    reduced.fit(
        d,
        duration_col="TIME_BY_DAYS",
        event_col="EVENT_ADconversion",
        strata=["BASE_DX"],
        formula=" + ".join(base_terms),
        robust=robust,
    )

    full = CoxPHFitter()
    full.fit(
        d,
        duration_col="TIME_BY_DAYS",
        event_col="EVENT_ADconversion",
        strata=["BASE_DX"],
        formula=" + ".join(union_terms),
        robust=robust,
    )
    reduced._analysis_label = label + " reference"
    full._analysis_label = label
    return reduced, full, d

def cox_term_row(cph, data, term, analysis, lr_reference=None):
    s = cph.summary.loc[term]
    row = {
        "analysis": analysis,
        "term": term,
        "n": len(data),
        "events": int(data["EVENT_ADconversion"].sum()),
        "HR": float(s["exp(coef)"]),
        "CI_low": float(s["exp(coef) lower 95%"]),
        "CI_high": float(s["exp(coef) upper 95%"]),
        "Wald_p": float(s["p"]),
    }
    if lr_reference is not None:
        chi2, p = lr_test(cph, lr_reference, df=1)
        row["LR_chi2"] = chi2
        row["LR_p"] = p
    return row
core = pd.read_excel(CORE_FILE, sheet_name=CORE_SHEET)
core["mean_alps_calc"] = (core.alps_L + core.alps_R) / 2
core["signed_alps_LminusR"] = core.alps_L - core.alps_R
core["absolute_alps_diff"] = core.signed_alps_LminusR.abs()
core["log_wmh"] = h_wmh(core.WMH)
tensor_stems = ["x_proj", "x_assoc", "y_proj", "z_assoc"]
for stem in tensor_stems:
    core[stem+"_absolute_diff"] = (core[stem+"_L"]-core[stem+"_R"]).abs()
for name in ["AGE_INDEX", "EDUCATION", "BASE_MMSE", "log_wmh", "BRAIN_PARENCHYMA_FRACTION", "alps_L", "alps_R", "mean_alps_calc", "signed_alps_LminusR", "absolute_alps_diff"] + [x+"_absolute_diff" for x in tensor_stems]:
    core[name+"_z"] = zscore(core[name])

core = core.rename(columns={"SEX(M = 0)": "sex_code"})

BASE_TERMS = [
    "AGE_INDEX_z", "sex_code", "EDUCATION_z", "BASE_MMSE_z",
    "log_wmh_z", "BRAIN_PARENCHYMA_FRACTION_z",
]
MEAN_ADJUSTED_TERMS = BASE_TERMS + ["mean_alps_calc_z"]


analysis1_rows = []

# Same reference cohort where possible
for term, label, base in [
    ("alps_L_z", "Left ALPS", BASE_TERMS),
    ("alps_R_z", "Right ALPS", BASE_TERMS),
    ("mean_alps_calc_z", "Mean ALPS", BASE_TERMS),
    ("signed_alps_LminusR_z", "Signed L-R difference", MEAN_ADJUSTED_TERMS),
    ("absolute_alps_diff_z", "Absolute |L-R| difference", MEAN_ADJUSTED_TERMS),
]:
    cols = ["TIME_BY_DAYS", "EVENT_ADconversion", "BASE_DX"] + list(dict.fromkeys(base + [term]))
    d = core[cols].dropna().copy()
    reduced, full, d1 = fit_nested_cox(d, base, [term], label)
    row = cox_term_row(full, d1, term, label, lr_reference=reduced)
    analysis1_rows.append(row)

analysis1 = pd.DataFrame(analysis1_rows)
display(analysis1)
save_table(analysis1, "03_left_right_signed_absolute_Cox.csv")


# Individual raw tensor components as specificity / negative-control panel
tensor_rows = []
for stem in tensor_stems:
    term = f"{stem}_absolute_diff_z"
    label = f"{stem} absolute L-R asymmetry"
    reduced, full, d1 = fit_nested_cox(core, MEAN_ADJUSTED_TERMS, [term], label)
    tensor_rows.append(cox_term_row(full, d1, term, label, lr_reference=reduced))

tensor_panel = pd.DataFrame(tensor_rows)
display(tensor_panel)
save_table(tensor_panel, "07_tensor_component_negative_controls.csv")

ai_raw = pd.to_numeric(core["absolute_alps_diff"], errors="coerce")
ai_mean = float(ai_raw.mean())
ai_sd = float(ai_raw.std(ddof=0))
core["absolute_ai_fixed_z"] = (ai_raw - ai_mean) / ai_sd

outlier_rows = []
for tail_pct in [0, 1, 2.5, 5]:
    if tail_pct == 0:
        d = core.copy()
        cutoff = np.nan
        label = "Full cohort"
    else:
        cutoff = float(core["absolute_alps_diff"].quantile(1 - tail_pct / 100))
        d = core.loc[core["absolute_alps_diff"] <= cutoff].copy()
        label = f"Exclude top {tail_pct}% Absolute AI"

    ref_terms = BASE_TERMS + ["mean_alps_calc_z"]
    reduced, full, d1 = fit_nested_cox(
        d, ref_terms, ["absolute_ai_fixed_z"], label
    )
    row = cox_term_row(full, d1, "absolute_ai_fixed_z", label, lr_reference=reduced)
    row["tail_cutoff_raw"] = cutoff
    outlier_rows.append(row)

outlier_sensitivity = pd.DataFrame(outlier_rows)
display(outlier_sensitivity)
save_table(outlier_sensitivity, "10_outlier_tail_exclusion_Cox.csv")

influence_rows = []
influence_status = []

primary_terms = MEAN_ADJUSTED_TERMS + ["absolute_alps_diff_z"]
primary_fit, primary_d = fit_stratified_cox(core, primary_terms, "Primary Absolute AI")
delta = primary_fit.compute_residuals(primary_d, kind="delta_beta")
if "absolute_alps_diff_z" not in delta.columns:
    raise KeyError("delta_beta residual does not contain absolute_alps_diff_z")

influence = delta["absolute_alps_diff_z"].abs().sort_values(ascending=False)
influence_table = pd.DataFrame({
    "row_index": influence.index,
    "abs_delta_beta_absoluteAI": influence.values,
})
save_table(influence_table.head(50), "11_top_influential_observations.csv")

for n_remove in [0, 1, 5, 10]:
    if n_remove == 0:
        d = core.copy()
        label = "No influence removal"
    else:
        remove_index = influence.head(n_remove).index
        d = core.drop(index=remove_index).copy()
        label = f"Remove top {n_remove} delta-beta observations"

    reduced, full, d1 = fit_nested_cox(
        d, MEAN_ADJUSTED_TERMS, ["absolute_alps_diff_z"], label
    )
    row = cox_term_row(
        full, d1, "absolute_alps_diff_z", label, lr_reference=reduced
    )
    influence_rows.append(row)

influence_result = pd.DataFrame(influence_rows)

display(influence_result)
save_table(influence_result, "12_influence_sensitivity_Cox.csv")
# Figure 4C: correlations of raw absolute component differences.
rows=[]
for stem in tensor_stems:
    v=core[["absolute_alps_diff",stem+"_absolute_diff"]].dropna()
    r,p=stats.spearmanr(v.iloc[:,0],v.iloc[:,1])
    rows.append(dict(component=stem,n=len(v),Spearman_rho=r,Spearman_p=p))
save_table(pd.DataFrame(rows),"Figure4C_component_correlations.csv")
