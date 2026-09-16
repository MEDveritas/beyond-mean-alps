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

SEED=20260830
N_BOOT=SETTINGS["competition_bootstrap"]
def zscore(series):
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd

def _raw_columns_from_terms(terms):
    cols = []
    for term in terms:
        term = term.strip()
        m = re.fullmatch(r"C\(([^)]+)\)", term)
        if m:
            cols.append(m.group(1).strip())
        else:
            cols.append(term)
    return list(dict.fromkeys(cols))

def fit_cox(data, terms, label, robust=True, penalizer=0.0):
    needed = ["TIME_BY_DAYS", "EVENT_ADconversion", "BASE_DX"] + _raw_columns_from_terms(terms)
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise KeyError(f"{label}: missing columns {missing}")
    d = data[needed].dropna().copy()

    cph = CoxPHFitter(penalizer=penalizer)
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

def lr_test(full_model, reduced_model, df=1):
    chi2 = 2.0 * (float(full_model.log_likelihood_) - float(reduced_model.log_likelihood_))
    p = float(stats.chi2.sf(max(chi2, 0.0), df))
    return chi2, p

def extract_hr(cph, term):
    s = cph.summary.loc[term]
    return {
        "HR": float(s["exp(coef)"]),
        "CI_low": float(s["exp(coef) lower 95%"]),
        "CI_high": float(s["exp(coef) upper 95%"]),
        "Wald_p": float(s["p"]),
        "coef": float(s["coef"]),
        "SE": float(s["se(coef)"]),
    }

def model_info(cph, d, model_name):
    return {
        "model": model_name,
        "n": len(d),
        "events": int(d["EVENT_ADconversion"].sum()),
        "parameters": int(len(cph.params_)),
        "log_likelihood": float(cph.log_likelihood_),
        "AIC_partial": float(cph.AIC_partial_),
        "apparent_concordance": float(cph.concordance_index_),
    }

core = pd.read_excel(CORE_FILE, sheet_name=CORE_SHEET)
core["PTID"] = core["PTID"].astype(str).str.strip()

required = [
    "PTID", "BASE_DX", "AGE_INDEX", "SEX(M = 0)", "EDUCATION", "BASE_MMSE",
    "EVENT_ADconversion", "TIME_BY_DAYS", "WMH", "BRAIN_PARENCHYMA_FRACTION",
    "alps_L", "alps_R", "ALPS",
    "x_proj_L", "x_assoc_L", "y_proj_L", "z_assoc_L",
    "x_proj_R", "x_assoc_R", "y_proj_R", "z_assoc_R",
]
missing = [c for c in required if c not in core.columns]
if missing:
    raise KeyError("Missing required columns:\n" + "\n".join(missing))

if core["PTID"].duplicated().any():
    raise ValueError("Core survival dataset must have one row per PTID.")

core = core.rename(columns={"SEX(M = 0)": "sex_code"})

core["log_wmh"] = np.log1p(pd.to_numeric(core["WMH"], errors="coerce"))
core["mean_alps"] = (core["alps_L"] + core["alps_R"]) / 2
core["signed_alps_diff"] = core["alps_L"] - core["alps_R"]
core["absolute_alps_diff"] = core["signed_alps_diff"].abs()

for stem in ["x_proj", "x_assoc", "y_proj", "z_assoc"]:
    core[f"{stem}_signed_diff"] = core[f"{stem}_L"] - core[f"{stem}_R"]
    core[f"{stem}_absolute_diff"] = core[f"{stem}_signed_diff"].abs()

to_z = [
    "AGE_INDEX", "EDUCATION", "BASE_MMSE", "log_wmh", "BRAIN_PARENCHYMA_FRACTION",
    "mean_alps", "absolute_alps_diff",
    "x_assoc_absolute_diff", "z_assoc_absolute_diff",
]
for c in to_z:
    core[f"{c}_z"] = zscore(core[c])

qc = pd.DataFrame({
    "item": [
        "participants", "events",
        "complete_absolute_AI", "complete_z_assoc_asymmetry",
        "complete_joint_AI_z_assoc"
    ],
    "value": [
        len(core),
        int(core["EVENT_ADconversion"].sum()),
        int(core["absolute_alps_diff_z"].notna().sum()),
        int(core["z_assoc_absolute_diff_z"].notna().sum()),
        int(core[["absolute_alps_diff_z", "z_assoc_absolute_diff_z"]].notna().all(axis=1).sum()),
    ]
})
display(qc)
save_table(qc, "01_core_QC.csv")

BASE_TERMS = [
    "AGE_INDEX_z",
    "sex_code",
    "EDUCATION_z",
    "BASE_MMSE_z",
    "log_wmh_z",
    "BRAIN_PARENCHYMA_FRACTION_z",
    "mean_alps_z",
]

AI_TERM = "absolute_alps_diff_z"
Z_TERM = "z_assoc_absolute_diff_z"
X_TERM = "x_assoc_absolute_diff_z"

model_terms = {
    "M0_reference": BASE_TERMS,
    "M_A_absoluteAI": BASE_TERMS + [AI_TERM],
    "M_Z_zAssoc": BASE_TERMS + [Z_TERM],
    "M_AZ_joint": BASE_TERMS + [AI_TERM, Z_TERM],
}

all_joint_terms = list(dict.fromkeys(BASE_TERMS + [AI_TERM, Z_TERM]))
needed = ["TIME_BY_DAYS", "EVENT_ADconversion", "BASE_DX"] + _raw_columns_from_terms(all_joint_terms)
primary_d = core[needed].dropna().copy()

fits = {}
model_rows = []
for name, terms in model_terms.items():
    cph, d = fit_cox(primary_d, terms, name, robust=True)
    fits[name] = cph
    model_rows.append(model_info(cph, d, name))

model_comparison = pd.DataFrame(model_rows)
display(model_comparison)
save_table(model_comparison, "02_primary_model_comparison.csv")

# Coefficients of the two competing asymmetry variables
coef_rows = []

for model_name in ["M_A_absoluteAI", "M_Z_zAssoc", "M_AZ_joint"]:
    cph = fits[model_name]
    for term, label in [
        (AI_TERM, "Absolute ALPS asymmetry"),
        (Z_TERM, "z_assoc absolute asymmetry"),
    ]:
        if term in cph.params_.index:
            r = extract_hr(cph, term)
            coef_rows.append({
                "model": model_name,
                "marker": label,
                "term": term,
                "n": len(primary_d),
                "events": int(primary_d["EVENT_ADconversion"].sum()),
                **r,
            })

primary_coefficients = pd.DataFrame(coef_rows)
display(primary_coefficients)
save_table(primary_coefficients, "03_primary_competition_coefficients.csv")

# Nested LR tests on exactly the same analytic sample
lr_rows = []

comparisons = [
    ("M_A_absoluteAI", "M0_reference", 1, "Add Absolute AI to reference"),
    ("M_Z_zAssoc", "M0_reference", 1, "Add z_assoc asymmetry to reference"),
    ("M_AZ_joint", "M_Z_zAssoc", 1, "Residual Absolute AI after z_assoc"),
    ("M_AZ_joint", "M_A_absoluteAI", 1, "Residual z_assoc after Absolute AI"),
    ("M_AZ_joint", "M0_reference", 2, "Joint 2-df contribution of both asymmetries"),
]

for full_name, reduced_name, df_diff, question in comparisons:
    chi2, p = lr_test(fits[full_name], fits[reduced_name], df=df_diff)
    lr_rows.append({
        "question": question,
        "full_model": full_name,
        "reduced_model": reduced_name,
        "df": df_diff,
        "LR_chi2": chi2,
        "LR_p": p,
    })

primary_lr = pd.DataFrame(lr_rows)
display(primary_lr)
save_table(primary_lr, "04_primary_nested_LR_tests.csv")

beta_ai_alone = float(fits["M_A_absoluteAI"].params_[AI_TERM])
beta_ai_joint = float(fits["M_AZ_joint"].params_[AI_TERM])

beta_z_alone = float(fits["M_Z_zAssoc"].params_[Z_TERM])
beta_z_joint = float(fits["M_AZ_joint"].params_[Z_TERM])

attenuation = pd.DataFrame([
    {
        "marker": "Absolute ALPS asymmetry",
        "beta_single_marker_model": beta_ai_alone,
        "beta_joint_model": beta_ai_joint,
        "relative_beta_attenuation": (
            1 - beta_ai_joint / beta_ai_alone if beta_ai_alone != 0 else np.nan
        ),
        "note": "Descriptive statistical attenuation; not mediation.",
    },
    {
        "marker": "z_assoc absolute asymmetry",
        "beta_single_marker_model": beta_z_alone,
        "beta_joint_model": beta_z_joint,
        "relative_beta_attenuation": (
            1 - beta_z_joint / beta_z_alone if beta_z_alone != 0 else np.nan
        ),
        "note": "Descriptive statistical attenuation; not mediation.",
    },
])

display(attenuation)
save_table(attenuation, "05_coefficient_attenuation.csv")

rng = np.random.default_rng(SEED)

boot_needed = [
    "TIME_BY_DAYS", "EVENT_ADconversion", "BASE_DX",
] + _raw_columns_from_terms(BASE_TERMS + [AI_TERM, Z_TERM])

boot_source = core[boot_needed].dropna().reset_index(drop=True).copy()

boot_rows = []
failed = 0

for b in range(N_BOOT):
    idx = rng.integers(0, len(boot_source), size=len(boot_source))
    db = boot_source.iloc[idx].copy().reset_index(drop=True)

    try:
        fit, _ = fit_cox(
            db,
            BASE_TERMS + [AI_TERM, Z_TERM],
            f"bootstrap_{b}",
            robust=False,
        )
        boot_rows.append({
            "replicate": b + 1,
            "beta_AI": float(fit.params_[AI_TERM]),
            "HR_AI": float(np.exp(fit.params_[AI_TERM])),
            "beta_zAssoc": float(fit.params_[Z_TERM]),
            "HR_zAssoc": float(np.exp(fit.params_[Z_TERM])),
        })
    except Exception:
        failed += 1

bootstrap_samples = pd.DataFrame(boot_rows)

def bootstrap_marker_summary(df, beta_col, hr_col, marker):
    if len(df) == 0:
        return {
            "marker": marker,
            "successful_replicates": 0,
            "median_HR": np.nan,
            "CI_low_HR": np.nan,
            "CI_high_HR": np.nan,
            "fraction_beta_positive": np.nan,
        }
    return {
        "marker": marker,
        "successful_replicates": len(df),
        "median_HR": float(df[hr_col].median()),
        "CI_low_HR": float(df[hr_col].quantile(0.025)),
        "CI_high_HR": float(df[hr_col].quantile(0.975)),
        "fraction_beta_positive": float((df[beta_col] > 0).mean()),
    }

bootstrap_summary = pd.DataFrame([
    bootstrap_marker_summary(
        bootstrap_samples, "beta_AI", "HR_AI", "Absolute ALPS asymmetry"
    ),
    bootstrap_marker_summary(
        bootstrap_samples, "beta_zAssoc", "HR_zAssoc", "z_assoc asymmetry"
    ),
])

if len(bootstrap_samples):
    bootstrap_summary["requested_replicates"] = N_BOOT
    bootstrap_summary["failed_replicates"] = failed
    both_positive = float(
        ((bootstrap_samples["beta_AI"] > 0) & (bootstrap_samples["beta_zAssoc"] > 0)).mean()
    )
else:
    both_positive = np.nan

bootstrap_joint_direction = pd.DataFrame([{
    "requested_replicates": N_BOOT,
    "successful_replicates": len(bootstrap_samples),
    "failed_replicates": failed,
    "fraction_both_betas_positive": both_positive,
}])

display(bootstrap_summary)
display(bootstrap_joint_direction)
save_table(bootstrap_samples, "16_bootstrap_joint_model_replicates.csv")
save_table(bootstrap_summary, "17_bootstrap_joint_model_summary.csv")
save_table(bootstrap_joint_direction, "18_bootstrap_joint_direction.csv")
if failed or len(bootstrap_samples) != N_BOOT:
    raise RuntimeError("Incomplete component-competition bootstrap")
