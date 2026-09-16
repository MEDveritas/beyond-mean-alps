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

from statsmodels.stats.multitest import multipletests
SEED=20260902
rng=np.random.default_rng(SEED)
BOOTSTRAP_B=SETTINGS["reorientation_bootstrap"]
INPUT_XLSX=DATA_DIR/"reoriented_clinical_alps.xlsx"
raw=pd.read_excel(INPUT_XLSX,sheet_name=SHEET)

d = raw.copy()

# Primary f1 measures.
d["F1_mean_alps"] = pd.to_numeric(d["ALPS"], errors="coerce")
d["F1_abs_asym"] = (
    pd.to_numeric(d["alps_L"], errors="coerce")
    - pd.to_numeric(d["alps_R"], errors="coerce")
).abs()
# f2 measures.
d["F2_mean_alps_calc"] = (
    pd.to_numeric(d["F2_alps_L"], errors="coerce")
    + pd.to_numeric(d["F2_alps_R"], errors="coerce")
) / 2
d["F2_abs_asym"] = (
    pd.to_numeric(d["F2_alps_L"], errors="coerce")
    - pd.to_numeric(d["F2_alps_R"], errors="coerce")
).abs()
# Internal f2 mean reconstruction check.
mean_err = (
    d.loc[d["F2_AVAILABLE"].eq(1), "F2_mean_alps_calc"]
    - pd.to_numeric(d.loc[d["F2_AVAILABLE"].eq(1), "F2_alps_mean"], errors="coerce")
).abs()

print("Max |recalculated f2 mean - server f2 mean|:", float(mean_err.max()))

matched = d.loc[pd.to_numeric(d["F2_AVAILABLE"], errors="coerce").eq(1)].copy()
matched = matched.dropna(subset=[
    "alps_L", "alps_R", "F1_mean_alps", "F1_abs_asym",
    "F2_alps_L", "F2_alps_R", "F2_mean_alps_calc", "F2_abs_asym"
]).reset_index(drop=True)

print("Matched f1/f2 analysis sample:", len(matched))

def icc_a1(x, y):
    # ICC(A,1): two-way absolute-agreement, single-measure ICC.
    # Here the two "raters" are f1 and f2 implementations.
    X = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n, k = X.shape
    grand = X.mean()
    subj_mean = X.mean(axis=1)
    rater_mean = X.mean(axis=0)

    ss_subject = k * np.sum((subj_mean - grand) ** 2)
    ss_rater = n * np.sum((rater_mean - grand) ** 2)
    ss_error = np.sum((X - subj_mean[:, None] - rater_mean[None, :] + grand) ** 2)

    ms_subject = ss_subject / (n - 1)
    ms_rater = ss_rater / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    return (
        (ms_subject - ms_error)
        / (
            ms_subject
            + (k - 1) * ms_error
            + k * (ms_rater - ms_error) / n
        )
    )

def agreement_row(label, x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = x.notna() & y.notna()
    x = x[ok].to_numpy(float)
    y = y[ok].to_numpy(float)
    diff = y - x
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    bias = np.mean(diff)
    return {
        "measure": label,
        "n": len(x),
        "f1_mean": np.mean(x),
        "f2_mean": np.mean(y),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "ICC_A1": icc_a1(x, y),
        "mean_bias_f2_minus_f1": bias,
    }

agreement = pd.DataFrame([
    agreement_row("Mean ALPS", matched["F1_mean_alps"], matched["F2_mean_alps_calc"]),
    agreement_row("Absolute ALPS asymmetry", matched["F1_abs_asym"], matched["F2_abs_asym"]),
])

agreement.to_csv(OUT / "02_f1_f2_reproducibility.csv", index=False)
agreement.round(5)

def zscore(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Cannot z-standardize a constant/non-finite variable.")
    return (s - s.mean()) / sd

def build_core_model_frame(df, mean_col, asym_col):
    out = pd.DataFrame(index=df.index)
    out["time_days"] = pd.to_numeric(df["TIME_BY_DAYS"], errors="coerce")
    out["event"] = pd.to_numeric(df["EVENT_ADconversion"], errors="coerce")
    out["strata"] = df["BASE_DX"].astype(str)
    out["sex_female"] = pd.to_numeric(df["SEX(M = 0)"], errors="coerce")

    wmh = pd.to_numeric(df["WMH"], errors="coerce")
    out["log_wmh_raw"] = h_wmh(wmh)

    out["age_z"] = zscore(df["AGE_INDEX"])
    out["education_z"] = zscore(df["EDUCATION"])
    out["mmse_z"] = zscore(df["BASE_MMSE"])
    out["log_wmh_z"] = zscore(out["log_wmh_raw"])
    out["bpf_z"] = zscore(df["BRAIN_PARENCHYMA_FRACTION"])
    out["mean_alps_z"] = zscore(df[mean_col])
    out["abs_asym_z"] = zscore(df[asym_col])

    cols = [
        "time_days", "event", "strata", "sex_female",
        "age_z", "education_z", "mmse_z", "log_wmh_z",
        "bpf_z", "mean_alps_z", "abs_asym_z"
    ]
    out = out[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    out = out.loc[
        out["time_days"].gt(0)
        & out["event"].isin([0, 1])
        & out["strata"].isin(["CN", "MCI"])
        & out["sex_female"].isin([0, 1])
    ].copy()
    out["event"] = out["event"].astype(int)
    return out.reset_index(drop=True)

CORE_REF = [
    "age_z", "sex_female", "education_z", "mmse_z",
    "log_wmh_z", "bpf_z", "mean_alps_z",
]
CORE_FULL = CORE_REF + ["abs_asym_z"]

def fit_stratified_cox(frame, covariates):
    use = frame[covariates + ["time_days", "event", "strata"]].copy()
    model = CoxPHFitter()
    model.fit(
        use,
        duration_col="time_days",
        event_col="event",
        strata=["strata"],
        robust=True,
        show_progress=False,
    )
    return model

def nested_lr(reference_fit, full_fit, df=1):
    chi2 = max(0.0, 2 * (full_fit.log_likelihood_ - reference_fit.log_likelihood_))
    p = stats.chi2.sf(chi2, df)
    return chi2, p

def asymmetry_result_row(label, frame):
    ref = fit_stratified_cox(frame, CORE_REF)
    full = fit_stratified_cox(frame, CORE_FULL)
    lr_chi2, lr_p = nested_lr(ref, full, 1)
    s = full.summary.loc["abs_asym_z"]
    return {
        "analysis": label,
        "n": len(frame),
        "events": int(frame["event"].sum()),
        "HR_per_1SD_abs_asym": float(s["exp(coef)"]),
        "CI_low": float(s["exp(coef) lower 95%"]),
        "CI_high": float(s["exp(coef) upper 95%"]),
        "Wald_p": float(s["p"]),
        "LR_chi2": float(lr_chi2),
        "LR_p": float(lr_p),
        "beta": float(s["coef"]),
        "SE": float(s["se(coef)"]),
    }, ref, full

f1_full_frame = build_core_model_frame(d, "F1_mean_alps", "F1_abs_asym")
f1_matched_frame = build_core_model_frame(matched, "F1_mean_alps", "F1_abs_asym")
f2_matched_frame = build_core_model_frame(matched, "F2_mean_alps_calc", "F2_abs_asym")

row_f1_full, f1_full_ref, f1_full_fit = asymmetry_result_row("f1_full_1131", f1_full_frame)
row_f1_matched, f1_match_ref, f1_match_fit = asymmetry_result_row("f1_matched", f1_matched_frame)
row_f2_matched, f2_match_ref, f2_match_fit = asymmetry_result_row("f2_matched", f2_matched_frame)

cox_comparison = pd.DataFrame([row_f1_full, row_f1_matched, row_f2_matched])
cox_comparison.to_csv(OUT / "03_f1_f2_same_specification_cox.csv", index=False)
cox_comparison.round(5)

# Build one paired raw dataset, then standardize once on the matched cohort.
paired_raw = matched.copy().reset_index(drop=True)

# Common covariates standardized once on the matched cohort.
paired_model = pd.DataFrame({
    "time_days": pd.to_numeric(paired_raw["TIME_BY_DAYS"], errors="coerce"),
    "event": pd.to_numeric(paired_raw["EVENT_ADconversion"], errors="coerce"),
    "strata": paired_raw["BASE_DX"].astype(str),
    "sex_female": pd.to_numeric(paired_raw["SEX(M = 0)"], errors="coerce"),
})
wmh = pd.to_numeric(paired_raw["WMH"], errors="coerce")
paired_model["log_wmh_raw"] = h_wmh(wmh)
paired_model["age_z"] = zscore(paired_raw["AGE_INDEX"])
paired_model["education_z"] = zscore(paired_raw["EDUCATION"])
paired_model["mmse_z"] = zscore(paired_raw["BASE_MMSE"])
paired_model["log_wmh_z"] = zscore(paired_model["log_wmh_raw"])
paired_model["bpf_z"] = zscore(paired_raw["BRAIN_PARENCHYMA_FRACTION"])

paired_model["f1_mean_z"] = zscore(paired_raw["F1_mean_alps"])
paired_model["f1_abs_z"] = zscore(paired_raw["F1_abs_asym"])
paired_model["f2_mean_z"] = zscore(paired_raw["F2_mean_alps_calc"])
paired_model["f2_abs_z"] = zscore(paired_raw["F2_abs_asym"])

paired_model = paired_model.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

def fit_pipeline_boot(frame, mean_z_col, abs_z_col):
    tmp = frame[[
        "time_days", "event", "strata", "age_z", "sex_female", "education_z",
        "mmse_z", "log_wmh_z", "bpf_z", mean_z_col, abs_z_col
    ]].copy()
    tmp = tmp.rename(columns={mean_z_col: "mean_alps_z", abs_z_col: "abs_asym_z"})
    fit = fit_stratified_cox(tmp, CORE_FULL)
    return float(fit.params_["abs_asym_z"])

boot_rows = []
n_pair = len(paired_model)

for b in range(BOOTSTRAP_B):
    idx = rng.integers(0, n_pair, size=n_pair)
    bs = paired_model.iloc[idx].reset_index(drop=True)
    try:
        beta_f1 = fit_pipeline_boot(bs, "f1_mean_z", "f1_abs_z")
        beta_f2 = fit_pipeline_boot(bs, "f2_mean_z", "f2_abs_z")
        boot_rows.append({
            "replicate": b + 1,
            "beta_f1": beta_f1,
            "beta_f2": beta_f2,
            "beta_difference_f2_minus_f1": beta_f2 - beta_f1,
            "HR_f1": np.exp(beta_f1),
            "HR_f2": np.exp(beta_f2),
        })
    except Exception:
        boot_rows.append({
            "replicate": b + 1,
            "beta_f1": np.nan,
            "beta_f2": np.nan,
            "beta_difference_f2_minus_f1": np.nan,
            "HR_f1": np.nan,
            "HR_f2": np.nan,
        })
    if (b + 1) % 100 == 0:
        print(f"Bootstrap {b+1}/{BOOTSTRAP_B}")

boot = pd.DataFrame(boot_rows)
boot_ok = boot.dropna()

summary_boot = pd.DataFrame([{
    "B_requested": BOOTSTRAP_B,
    "B_successful": len(boot_ok),
    "median_HR_f1": boot_ok["HR_f1"].median(),
    "HR_f1_CI_low": boot_ok["HR_f1"].quantile(.025),
    "HR_f1_CI_high": boot_ok["HR_f1"].quantile(.975),
    "median_HR_f2": boot_ok["HR_f2"].median(),
    "HR_f2_CI_low": boot_ok["HR_f2"].quantile(.025),
    "HR_f2_CI_high": boot_ok["HR_f2"].quantile(.975),
    "median_beta_difference_f2_minus_f1": boot_ok["beta_difference_f2_minus_f1"].median(),
    "beta_difference_CI_low": boot_ok["beta_difference_f2_minus_f1"].quantile(.025),
    "beta_difference_CI_high": boot_ok["beta_difference_f2_minus_f1"].quantile(.975),
    "same_positive_direction_fraction": np.mean(
        (boot_ok["beta_f1"] > 0) & (boot_ok["beta_f2"] > 0)
    ),
}])

boot.to_csv(OUT / "04_f1_f2_paired_bootstrap_replicates.csv", index=False)
summary_boot.to_csv(OUT / "04_f1_f2_paired_bootstrap_summary.csv", index=False)
summary_boot.round(5)

scalar = matched.copy()

# Hemisphere-composite FA and MD across the same two ALPS ROIs.
scalar["FA_L_composite"] = (
    pd.to_numeric(scalar["F2_FA_proj_L"], errors="coerce")
    + pd.to_numeric(scalar["F2_FA_assoc_L"], errors="coerce")
) / 2
scalar["FA_R_composite"] = (
    pd.to_numeric(scalar["F2_FA_proj_R"], errors="coerce")
    + pd.to_numeric(scalar["F2_FA_assoc_R"], errors="coerce")
) / 2
scalar["MD_L_composite"] = (
    pd.to_numeric(scalar["F2_MD_proj_L"], errors="coerce")
    + pd.to_numeric(scalar["F2_MD_assoc_L"], errors="coerce")
) / 2
scalar["MD_R_composite"] = (
    pd.to_numeric(scalar["F2_MD_proj_R"], errors="coerce")
    + pd.to_numeric(scalar["F2_MD_assoc_R"], errors="coerce")
) / 2

scalar["FA_composite_abs"] = (scalar["FA_L_composite"] - scalar["FA_R_composite"]).abs()
scalar["MD_composite_abs"] = (scalar["MD_L_composite"] - scalar["MD_R_composite"]).abs()

scalar["FA_proj_abs"] = (
    pd.to_numeric(scalar["F2_FA_proj_L"], errors="coerce")
    - pd.to_numeric(scalar["F2_FA_proj_R"], errors="coerce")
).abs()
scalar["FA_assoc_abs"] = (
    pd.to_numeric(scalar["F2_FA_assoc_L"], errors="coerce")
    - pd.to_numeric(scalar["F2_FA_assoc_R"], errors="coerce")
).abs()
scalar["MD_proj_abs"] = (
    pd.to_numeric(scalar["F2_MD_proj_L"], errors="coerce")
    - pd.to_numeric(scalar["F2_MD_proj_R"], errors="coerce")
).abs()
scalar["MD_assoc_abs"] = (
    pd.to_numeric(scalar["F2_MD_assoc_L"], errors="coerce")
    - pd.to_numeric(scalar["F2_MD_assoc_R"], errors="coerce")
).abs()

scalar_vars = [
    "FA_composite_abs", "MD_composite_abs",
    "FA_proj_abs", "FA_assoc_abs", "MD_proj_abs", "MD_assoc_abs",
]


def build_scalar_reference_frame(df):
    out = pd.DataFrame(index=df.index)
    out["time_days"] = pd.to_numeric(df["TIME_BY_DAYS"], errors="coerce")
    out["event"] = pd.to_numeric(df["EVENT_ADconversion"], errors="coerce")
    out["strata"] = df["BASE_DX"].astype(str)
    out["sex_female"] = pd.to_numeric(df["SEX(M = 0)"], errors="coerce")

    wmh = pd.to_numeric(df["WMH"], errors="coerce")
    out["log_wmh_raw"] = h_wmh(wmh)

    out["age_z"] = zscore(df["AGE_INDEX"])
    out["education_z"] = zscore(df["EDUCATION"])
    out["mmse_z"] = zscore(df["BASE_MMSE"])
    out["log_wmh_z"] = zscore(out["log_wmh_raw"])
    out["bpf_z"] = zscore(df["BRAIN_PARENCHYMA_FRACTION"])
    out["mean_alps_z"] = zscore(df["F2_mean_alps_calc"])
    out["alps_abs_z"] = zscore(df["F2_abs_asym"])

    for var in scalar_vars:
        out[var + "_z"] = zscore(df[var])

    return out.replace([np.inf, -np.inf], np.nan)

SCALAR_REF = [
    "age_z", "sex_female", "education_z", "mmse_z",
    "log_wmh_z", "bpf_z", "mean_alps_z",
]

scalar_frame = build_scalar_reference_frame(scalar)

def scalar_test(var):
    zvar = var + "_z"
    use_cols = SCALAR_REF + [zvar] + ["time_days", "event", "strata"]
    use = scalar_frame[use_cols].dropna().reset_index(drop=True)

    ref = fit_stratified_cox(use, SCALAR_REF)
    full = fit_stratified_cox(use, SCALAR_REF + [zvar])
    lr_chi2, lr_p = nested_lr(ref, full, 1)
    s = full.summary.loc[zvar]

    return {
        "scalar_asymmetry": var,
        "n": len(use),
        "events": int(use["event"].sum()),
        "HR_per_1SD": float(s["exp(coef)"]),
        "CI_low": float(s["exp(coef) lower 95%"]),
        "CI_high": float(s["exp(coef) upper 95%"]),
        "Wald_p": float(s["p"]),
        "LR_chi2": float(lr_chi2),
        "LR_p": float(lr_p),
    }

scalar_models = pd.DataFrame([scalar_test(v) for v in scalar_vars])
scalar_models["FDR_q_Wald"] = multipletests(
    scalar_models["Wald_p"].to_numpy(float), method="fdr_bh"
)[1]
scalar_models["FDR_q_LR"] = multipletests(
    scalar_models["LR_p"].to_numpy(float), method="fdr_bh"
)[1]

scalar_models.to_csv(OUT / "06_FA_MD_negative_control_cox.csv", index=False)
scalar_models.round(5)
if len(boot_ok) != BOOTSTRAP_B:
    raise RuntimeError("Incomplete paired-method bootstrap; inspect replicate CSV")
