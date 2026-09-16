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

SEED=20260902
EXPECTED_N=1131
EXPECTED_EVENTS=115
STRICT_FULL_TECH_COVERAGE=True
ANALYTIC_XLSX=DATA_DIR/"reoriented_clinical_alps.xlsx"
MOTION_CSV=DATA_DIR/"diffusion_motion_metrics.csv"
ORIENTATION_CSV=DATA_DIR/"head_orientation_angles.csv"
ACQ_PLANE_CSV=DATA_DIR/"acquisition_plane_angles.csv"

SHEET = "survival_dataset_final_age"

analytic = pd.read_excel(ANALYTIC_XLSX, sheet_name=SHEET)
analytic["PTID"] = analytic["PTID"].astype(str).str.strip()

required_core = [
    "PTID", "BASE_DX", "AGE_INDEX", "SEX(M = 0)", "EDUCATION", "BASE_MMSE",
    "EVENT_ADconversion", "TIME_BY_DAYS", "WMH", "BRAIN_PARENCHYMA_FRACTION",
    "SCANNER_MFR", "DIRECTIONS", "THICKNESS", "TR", "TE",
    "alps_L", "alps_R", "ALPS",
]
missing = [c for c in required_core if c not in analytic.columns]
if missing:
    raise ValueError(f"Missing required analytic columns: {missing}")

if analytic["PTID"].duplicated().any():
    raise ValueError("Duplicate PTIDs found in analytic workbook.")

n = len(analytic)
events = int(pd.to_numeric(analytic["EVENT_ADconversion"], errors="coerce").sum())

print(f"Analytic cohort: N={n}, events={events}")

if n != EXPECTED_N or events != EXPECTED_EVENTS:
    raise ValueError(
        f"Canonical cohort mismatch. Expected {EXPECTED_N}/{EXPECTED_EVENTS}, got {n}/{events}."
    )

motion = pd.read_csv(MOTION_CSV)
orientation = pd.read_csv(ORIENTATION_CSV)

for frame, name in [(motion, "motion"), (orientation, "orientation")]:
    if "PTID" not in frame.columns:
        raise ValueError(f"{name} file has no PTID column.")
    frame["PTID"] = (
        frame["PTID"].astype(str).str.strip()
        .str.replace("_ID_", "_S_", regex=False)
    )
    if frame["PTID"].duplicated().any():
        dup = frame.loc[frame["PTID"].duplicated(keep=False), "PTID"].unique().tolist()
        raise ValueError(f"Duplicate PTIDs remain in {name} CSV: {dup[:20]}")

# Keep only the fields relevant to this analysis.
motion_keep = [
    c for c in [
        "PTID", "mean_relative_rms", "mean_restricted_relative_rms",
        "p95_relative_rms", "outlier_slice_fraction",
        "motion_success", "motion_failure_reason", "duplicate_conflict"
    ] if c in motion.columns
]
orientation_keep = [
    c for c in [
        "PTID", "pitch_deg", "yaw_deg", "roll_deg",
        "orientation_success", "orientation_failure_reason", "duplicate_conflict"
    ] if c in orientation.columns
]

d = analytic.merge(motion[motion_keep], on="PTID", how="left", validate="one_to_one")
d = d.merge(
    orientation[orientation_keep],
    on="PTID",
    how="left",
    validate="one_to_one",
    suffixes=("_motion", "_orientation"),
)

# Respect explicit QC success flags if they exist.
if "motion_success" in d.columns:
    fail = ~d["motion_success"].fillna(False).astype(bool)
    for c in ["mean_relative_rms", "mean_restricted_relative_rms", "p95_relative_rms", "outlier_slice_fraction"]:
        if c in d.columns:
            d.loc[fail, c] = np.nan

if "orientation_success" in d.columns:
    fail = ~d["orientation_success"].fillna(False).astype(bool)
    for c in ["pitch_deg", "yaw_deg", "roll_deg"]:
        if c in d.columns:
            d.loc[fail, c] = np.nan

motion_available = d["mean_relative_rms"].notna()
orientation_available = d[["pitch_deg", "yaw_deg", "roll_deg"]].notna().all(axis=1)

coverage = pd.DataFrame({
    "domain": ["motion", "head_orientation", "motion_and_orientation"],
    "N_available": [
        int(motion_available.sum()),
        int(orientation_available.sum()),
        int((motion_available & orientation_available).sum()),
    ],
    "events_available": [
        int(d.loc[motion_available, "EVENT_ADconversion"].sum()),
        int(d.loc[orientation_available, "EVENT_ADconversion"].sum()),
        int(d.loc[motion_available & orientation_available, "EVENT_ADconversion"].sum()),
    ],
})
coverage["N_missing"] = EXPECTED_N - coverage["N_available"]
coverage["events_missing"] = EXPECTED_EVENTS - coverage["events_available"]

display(coverage)
coverage.to_csv(OUT / "01_technical_covariate_coverage.csv", index=False)

missing_tech = d.loc[
    ~(motion_available & orientation_available),
    ["PTID", "EVENT_ADconversion", "mean_relative_rms", "pitch_deg", "yaw_deg", "roll_deg"]
].copy()

if len(missing_tech):
    print("\nParticipants still missing motion/orientation:")
    display(missing_tech)
    missing_tech.to_csv(OUT / "01_missing_motion_orientation_ids.csv", index=False)
else:
    print("\nAll 1,131 analytic participants have motion and head-orientation covariates.")

if STRICT_FULL_TECH_COVERAGE and len(missing_tech):
    raise ValueError(
        "STRICT_FULL_TECH_COVERAGE=True, but some analytic participants still lack "
        "motion/orientation values. Supply complete measurements for all participants."
    )

plane_available = False
plane_covariates = []

if ACQ_PLANE_CSV is None:
    print("PENDING: acquisition-plane CSV not found.")
else:
    plane = pd.read_csv(ACQ_PLANE_CSV)
    if "PTID" not in plane.columns:
        raise ValueError("Acquisition-plane CSV has no PTID column.")
    plane["PTID"] = (
        plane["PTID"].astype(str).str.strip()
        .str.replace("_ID_", "_S_", regex=False)
    )
    if plane["PTID"].duplicated().any():
        dup = plane.loc[plane["PTID"].duplicated(keep=False), "PTID"].unique().tolist()
        raise ValueError(f"Duplicate PTIDs in acquisition-plane CSV: {dup[:20]}")

    plane_use = plane[["PTID", "acq_tilt_x_deg", "acq_tilt_y_deg"]].copy()

    d = d.merge(plane_use, on="PTID", how="left", validate="one_to_one")
    plane_covariates = ["acq_tilt_x_deg", "acq_tilt_y_deg"]
    plane_complete = d[plane_covariates].notna().all(axis=1)

    print(
        f"Acquisition-plane coverage: {int(plane_complete.sum())}/{EXPECTED_N}, "
        f"events={int(d.loc[plane_complete, 'EVENT_ADconversion'].sum())}/{EXPECTED_EVENTS}"
    )
    if plane_complete.all():
        plane_available = True
    else:
        warnings.warn(
            "Acquisition-plane file is present but incomplete. "
            "Plane-specific models will use a same-sample complete-case comparison."
        )
        plane_available = True

    plane_use.to_csv(OUT / "02_acquisition_plane_derived.csv", index=False)

def numeric(s):
    return pd.to_numeric(s, errors="coerce")

d["alps_L_num"] = numeric(d["alps_L"])
d["alps_R_num"] = numeric(d["alps_R"])
d["mean_alps"] = numeric(d["ALPS"])

d["abs_asym_raw"] = (d["alps_L_num"] - d["alps_R_num"]).abs()
d["abs_asym_relative"] = (
    2 * d["abs_asym_raw"] / (d["alps_L_num"] + d["alps_R_num"])
)
d["abs_asym_logratio"] = np.abs(np.log(d["alps_L_num"] / d["alps_R_num"]))

d["log_wmh"] = h_wmh(d["WMH"])

derived_summary = d[
    ["mean_alps", "abs_asym_raw", "abs_asym_relative", "abs_asym_logratio"]
].describe().T
display(derived_summary)
derived_summary.to_csv(OUT / "03_asymmetry_definition_descriptives.csv")

def zscore_full(s):
    s = numeric(s).astype(float)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd

# Core model variables
d["time_days"] = numeric(d["TIME_BY_DAYS"])
d["event"] = numeric(d["EVENT_ADconversion"])
d["strata"] = d["BASE_DX"].astype(str)
d["sex_female"] = numeric(d["SEX(M = 0)"])

d["age_z"] = zscore_full(d["AGE_INDEX"])
d["education_z"] = zscore_full(d["EDUCATION"])
d["mmse_z"] = zscore_full(d["BASE_MMSE"])
d["log_wmh_z"] = zscore_full(d["log_wmh"])
d["bpf_z"] = zscore_full(d["BRAIN_PARENCHYMA_FRACTION"])
d["mean_alps_z"] = zscore_full(d["mean_alps"])

for c in ["abs_asym_raw", "abs_asym_relative", "abs_asym_logratio"]:
    d[c + "_z"] = zscore_full(d[c])

# Protocol
d["directions_z"] = zscore_full(d["DIRECTIONS"])
d["thickness_z"] = zscore_full(d["THICKNESS"])
d["tr_z"] = zscore_full(d["TR"])
d["te_z"] = zscore_full(d["TE"])

# Motion
for c in [
    "mean_relative_rms",
    "mean_restricted_relative_rms",
    "p95_relative_rms",
    "outlier_slice_fraction",
]:
    if c in d.columns:
        d[c + "_z"] = zscore_full(d[c])

# Head orientation
for c in ["pitch_deg", "yaw_deg", "roll_deg"]:
    d[c + "_z"] = zscore_full(d[c])

# Acquisition-plane orientation
if plane_available:
    for c in plane_covariates:
        d[c + "_z"] = zscore_full(d[c])

# Replace existing manufacturer indicators before encoding.
existing_vendor_cols = [c for c in d.columns if str(c).startswith("vendor_")]
if existing_vendor_cols:
    d = d.drop(columns=existing_vendor_cols)
vendor = (
    d["SCANNER_MFR"].astype(str).str.strip().str.upper()
    .replace({"GE MEDICAL SYSTEMS": "GE", "GENERAL ELECTRIC": "GE"})
)
vendor_dummies = pd.get_dummies(vendor, prefix="vendor", drop_first=True, dtype=float)
d = pd.concat([d, vendor_dummies], axis=1)
vendor_cols = list(vendor_dummies.columns)

print("Scanner dummy variables:", vendor_cols)

CORE_NO_ASYM = [
    "age_z", "sex_female", "education_z", "mmse_z",
    "log_wmh_z", "bpf_z", "mean_alps_z",
]
PRIMARY_ASYM = "abs_asym_raw_z"
CORE_WITH_ASYM = CORE_NO_ASYM + [PRIMARY_ASYM]

PROTOCOL_BLOCK = vendor_cols + [
    "directions_z", "thickness_z", "tr_z", "te_z"
]
MOTION_BLOCK = ["mean_relative_rms_z"]
HEAD_ORIENTATION_BLOCK = ["pitch_deg_z", "yaw_deg_z", "roll_deg_z"]
PLANE_BLOCK = [c + "_z" for c in plane_covariates] if plane_available else []

print("Core covariates:", CORE_WITH_ASYM)
print("Protocol block:", PROTOCOL_BLOCK)
print("Motion block:", MOTION_BLOCK)
print("Head-orientation block:", HEAD_ORIENTATION_BLOCK)
print("Acquisition-plane block:", PLANE_BLOCK if PLANE_BLOCK else "PENDING")

def valid_model_rows(df, covariates):
    # Use unique column names when selecting model inputs.
    source = df.loc[:, ~df.columns.duplicated(keep="last")].copy()
    cols = list(dict.fromkeys(covariates + ["time_days", "event", "strata"]))
    use = source[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    use = use.loc[
        use["time_days"].gt(0)
        & use["event"].isin([0, 1])
        & use["strata"].isin(["CN", "MCI"])
    ].copy()
    use["event"] = use["event"].astype(int)
    return use.reset_index(drop=True)

def fit_cox(frame, covariates):
    # A dummy variable can become all-zero after complete-case filtering
    # (e.g. vendor_PHILIPS in the acquisition-plane subset). lifelines
    # normalizes such columns to NaN, so exclude them before fitting.
    used_covariates = [c for c in covariates if frame[c].nunique(dropna=True) > 1]
    dropped = [c for c in covariates if c not in used_covariates]
    if dropped:
        print(f"Dropped non-varying Cox covariates: {dropped}")
    model = CoxPHFitter()
    model.fit(
        frame[used_covariates + ["time_days", "event", "strata"]],
        duration_col="time_days",
        event_col="event",
        strata=["strata"],
        robust=True,
        show_progress=False,
    )
    model.alps_used_covariates_ = used_covariates
    model.alps_dropped_nonvarying_ = dropped
    return model

def lr_test(reduced, full, df_added):
    if df_added <= 0:
        return 0.0, np.nan
    chi2 = max(0.0, 2 * (full.log_likelihood_ - reduced.log_likelihood_))
    p = stats.chi2.sf(chi2, df_added)
    return chi2, p

def same_sample_nested_comparison(df, added_covariates, label, asym_var=PRIMARY_ASYM):
    ref_covs = [c if c != PRIMARY_ASYM else asym_var for c in CORE_WITH_ASYM]
    full_covs = ref_covs + added_covariates

    # The expanded model determines the complete-case sample.
    use = valid_model_rows(df, full_covs)
    ref = fit_cox(use, ref_covs)
    full = fit_cox(use, full_covs)

    estimated_added = [c for c in added_covariates if c in full.params_.index]
    lr_chi2, lr_p = lr_test(ref, full, len(estimated_added))

    s0 = ref.summary.loc[asym_var]
    s1 = full.summary.loc[asym_var]
    beta0 = float(s0["coef"])
    beta1 = float(s1["coef"])

    attenuation = (
        100 * (abs(beta0) - abs(beta1)) / abs(beta0)
        if beta0 != 0 else np.nan
    )

    return {
        "model": label,
        "N": len(use),
        "events": int(use["event"].sum()),
        "asymmetry_variable": asym_var,
        "reference_HR": float(s0["exp(coef)"]),
        "reference_CI_low": float(s0["exp(coef) lower 95%"]),
        "reference_CI_high": float(s0["exp(coef) upper 95%"]),
        "adjusted_HR": float(s1["exp(coef)"]),
        "adjusted_CI_low": float(s1["exp(coef) lower 95%"]),
        "adjusted_CI_high": float(s1["exp(coef) upper 95%"]),
        "adjusted_p": float(s1["p"]),
        "reference_beta": beta0,
        "adjusted_beta": beta1,
        "abs_beta_attenuation_percent": attenuation,
        "LR_chi2_added_block": lr_chi2,
        "LR_df": len(estimated_added),
        "dropped_nonvarying_covariates": ",".join(full.alps_dropped_nonvarying_),
        "LR_p_added_block": lr_p,
    }, use, ref, full

technical_specs = [
    ("Scanner/protocol", PROTOCOL_BLOCK),
    ("Within-scan motion", MOTION_BLOCK),
    ("Head orientation", HEAD_ORIENTATION_BLOCK),
    ("Motion + head orientation", MOTION_BLOCK + HEAD_ORIENTATION_BLOCK),
    (
        "All technical except acquisition plane",
        PROTOCOL_BLOCK + MOTION_BLOCK + HEAD_ORIENTATION_BLOCK
    ),
]

if plane_available:
    technical_specs.insert(3, ("Acquisition-plane orientation", PLANE_BLOCK))
    technical_specs.append((
        "All technical covariates",
        PROTOCOL_BLOCK + PLANE_BLOCK + MOTION_BLOCK + HEAD_ORIENTATION_BLOCK
    ))

technical_rows = []
technical_fits = {}

for label, added in technical_specs:
    try:
        row, use, ref, full = same_sample_nested_comparison(d, added, label)
        technical_rows.append(row)
        technical_fits[label] = {"use": use, "ref": ref, "full": full, "added": added}
    except Exception as exc:
        warnings.warn(f"{label} failed: {exc}")

technical_table = pd.DataFrame(technical_rows)
display(technical_table.round(5))
technical_table.to_csv(OUT / "04_technical_cox_model_comparison.csv", index=False)

motion_sensitivity_vars = [
    ("Mean relative RMS (primary)", "mean_relative_rms_z"),
    ("Mean restricted relative RMS", "mean_restricted_relative_rms_z"),
    ("P95 relative RMS", "p95_relative_rms_z"),
    ("Outlier-slice fraction", "outlier_slice_fraction_z"),
]

motion_rows = []
for label, var in motion_sensitivity_vars:
    if var not in d.columns:
        continue
    try:
        row, _, _, _ = same_sample_nested_comparison(
            d, [var], f"Motion sensitivity: {label}"
        )
        motion_rows.append(row)
    except Exception as exc:
        warnings.warn(f"{label} failed: {exc}")

motion_sensitivity_table = pd.DataFrame(motion_rows)
display(motion_sensitivity_table.round(5))
motion_sensitivity_table.to_csv(OUT / "05_motion_sensitivity_cox.csv", index=False)

ASYM_DEFS = {
    "Raw absolute difference |L-R|": "abs_asym_raw_z",
    "Relative absolute asymmetry 2|L-R|/(L+R)": "abs_asym_relative_z",
    "Absolute log-ratio |log(L/R)|": "abs_asym_logratio_z",
}

def asymmetry_definition_test(df, asym_var, tech_covs=None, label="core"):
    df = df.loc[:, ~df.columns.duplicated(keep="last")].copy()
    tech_covs = tech_covs or []
    base_covs = CORE_NO_ASYM + tech_covs
    full_covs = base_covs + [asym_var]

    use = valid_model_rows(df, full_covs)
    # Exclude nonvarying covariates in the complete-case sample.
    base_covs = [c for c in base_covs if use[c].nunique(dropna=True) > 1]
    if use[asym_var].nunique(dropna=True) <= 1:
        raise ValueError(f"Asymmetry variable has no variation: {asym_var}")
    full_covs = base_covs + [asym_var]
    ref = fit_cox(use, base_covs)
    full = fit_cox(use, full_covs)
    lr_chi2, lr_p = lr_test(ref, full, 1)

    s = full.summary.loc[asym_var]
    return {
        "adjustment": label,
        "asymmetry_variable": asym_var,
        "N": len(use),
        "events": int(use["event"].sum()),
        "HR_per_1SD": float(s["exp(coef)"]),
        "CI_low": float(s["exp(coef) lower 95%"]),
        "CI_high": float(s["exp(coef) upper 95%"]),
        "Wald_p": float(s["p"]),
        "LR_chi2": lr_chi2,
        "LR_p": lr_p,
    }

alt_rows = []

# Common core sample across all three definitions
common_core_covs = CORE_NO_ASYM + list(ASYM_DEFS.values())
common_core = valid_model_rows(d, common_core_covs)

for label, var in ASYM_DEFS.items():
    result = asymmetry_definition_test(common_core, var, tech_covs=[], label="Core adjusted")
    result["definition"] = label
    alt_rows.append(result)

if plane_available:
    all_tech = PROTOCOL_BLOCK + PLANE_BLOCK + MOTION_BLOCK + HEAD_ORIENTATION_BLOCK
    common_all_covs = CORE_NO_ASYM + all_tech + list(ASYM_DEFS.values())
    common_all = valid_model_rows(d, common_all_covs)

    for label, var in ASYM_DEFS.items():
        result = asymmetry_definition_test(
            common_all,
            var,
            tech_covs=all_tech,
            label="All-technical adjusted",
        )
        result["definition"] = label
        alt_rows.append(result)

alternative_table = pd.DataFrame(alt_rows)
display(alternative_table.round(5))
alternative_table.to_csv(OUT / "06_alternative_asymmetry_definitions.csv", index=False)

# ============================================================
# Adjusted 36/60-month absolute risks and risk differences
# ============================================================

RISK_BOOTSTRAP_B = SETTINGS["risk_bootstrap"]
RISK_SEED = 20260902
risk_rng = np.random.default_rng(RISK_SEED)

MONTH_DAYS = 365.25 / 12
RISK_TIMES_DAYS = {
    "36_month": 36 * MONTH_DAYS,
    "60_month": 60 * MONTH_DAYS,
}

# Primary f1 model on the canonical core complete-case sample.
# Use the canonical primary model's full-cohort scale for primary absolute risk.
risk_d = d.copy()
for target, raw_col in {'age_z':'AGE_INDEX','education_z':'EDUCATION','mmse_z':'BASE_MMSE','log_wmh_z':'log_wmh','bpf_z':'BRAIN_PARENCHYMA_FRACTION','mean_alps_z':'mean_alps','abs_asym_raw_z':'abs_asym_raw'}.items():
    values=pd.to_numeric(risk_d[raw_col],errors='coerce')
    risk_d[target]=(values-values.mean())/values.std(ddof=0)
risk_use = valid_model_rows(risk_d, CORE_WITH_ASYM)
risk_model = fit_cox(risk_use, CORE_WITH_ASYM)

print(
    f"Absolute-risk model sample: N={len(risk_use)}, "
    f"events={int(risk_use['event'].sum())}"
)

def _prediction_frame(frame, covariates):
    return frame[covariates + ["strata"]].copy()

def standardized_conversion_risk(model, target_frame, asym_value_z, times_dict):
    # Marginal standardization:
    # keep observed covariates/BASE_DX, set asymmetry to one common level,
    # predict S(t), then average 1-S(t) across the target population.
    pred = _prediction_frame(target_frame, CORE_WITH_ASYM)
    pred[PRIMARY_ASYM] = float(asym_value_z)

    times = list(times_dict.values())
    sf = model.predict_survival_function(pred, times=times)

    out = {}
    sf_times = sf.index.to_numpy(dtype=float)
    for label, t in times_dict.items():
        idx = np.argmin(np.abs(sf_times - float(t)))
        mean_survival = float(sf.iloc[idx].mean())
        out[label] = 1.0 - mean_survival
    return out

PRIMARY_RISK_LEVELS = {
    "0SD": 0.0,
    "plus1SD": 1.0,
}

def point_estimate_risk_table(model, target_frame, level_dict, contrast_name):
    estimates = {
        level: standardized_conversion_risk(model, target_frame, z, RISK_TIMES_DAYS)
        for level, z in level_dict.items()
    }
    levels = list(level_dict.keys())
    low_name, high_name = levels[0], levels[1]

    rows = []
    for time_label in RISK_TIMES_DAYS:
        low_risk = estimates[low_name][time_label]
        high_risk = estimates[high_name][time_label]
        rows.append({
            "contrast": contrast_name,
            "time": time_label,
            "lower_asymmetry_level": low_name,
            "lower_asymmetry_z": level_dict[low_name],
            "lower_adjusted_risk": low_risk,
            "higher_asymmetry_level": high_name,
            "higher_asymmetry_z": level_dict[high_name],
            "higher_adjusted_risk": high_risk,
            "absolute_risk_difference": high_risk - low_risk,
        })
    return pd.DataFrame(rows)

risk_point_primary = point_estimate_risk_table(
    risk_model, risk_use, PRIMARY_RISK_LEVELS, "0 SD vs +1 SD"
)
risk_point = risk_point_primary

display(
    risk_point.assign(
        lower_adjusted_risk_percent=100 * risk_point["lower_adjusted_risk"],
        higher_adjusted_risk_percent=100 * risk_point["higher_adjusted_risk"],
        absolute_risk_difference_percent_points=100 * risk_point["absolute_risk_difference"],
    ).round(3)
)

# ----------------------------
# Participant-level bootstrap
# ----------------------------

bootstrap_target = risk_use.copy().reset_index(drop=True)
n_risk = len(bootstrap_target)
boot_rows = []

for b in range(RISK_BOOTSTRAP_B):
    idx = risk_rng.integers(0, n_risk, size=n_risk)
    bs = bootstrap_target.iloc[idx].reset_index(drop=True)

    try:
        bs_model = fit_cox(bs, CORE_WITH_ASYM)

        for contrast_name, level_dict in [
            ("0 SD vs +1 SD", PRIMARY_RISK_LEVELS),
        ]:
            levels = list(level_dict.keys())
            low_name, high_name = levels[0], levels[1]

            low = standardized_conversion_risk(
                bs_model, bootstrap_target, level_dict[low_name], RISK_TIMES_DAYS
            )
            high = standardized_conversion_risk(
                bs_model, bootstrap_target, level_dict[high_name], RISK_TIMES_DAYS
            )

            for time_label in RISK_TIMES_DAYS:
                boot_rows.append({
                    "replicate": b + 1,
                    "contrast": contrast_name,
                    "time": time_label,
                    "lower_risk": low[time_label],
                    "higher_risk": high[time_label],
                    "risk_difference": high[time_label] - low[time_label],
                })

    except Exception:
        boot_rows.append({
            "replicate": b + 1,
            "contrast": "__FAILED__",
            "time": "__FAILED__",
            "lower_risk": np.nan,
            "higher_risk": np.nan,
            "risk_difference": np.nan,
        })

    if (b + 1) % 50 == 0:
        print(f"Risk bootstrap {b+1}/{RISK_BOOTSTRAP_B}")

risk_boot = pd.DataFrame(boot_rows)
risk_boot_ok = risk_boot.loc[risk_boot["contrast"] != "__FAILED__"].dropna().copy()

def percentile_ci(x):
    x = pd.Series(x).dropna().astype(float)
    return x.quantile(0.025), x.quantile(0.975)

summary_rows = []

for _, point_row in risk_point.iterrows():
    subset = risk_boot_ok.loc[
        (risk_boot_ok["contrast"] == point_row["contrast"])
        & (risk_boot_ok["time"] == point_row["time"])
    ]

    low_lo, low_hi = percentile_ci(subset["lower_risk"])
    high_lo, high_hi = percentile_ci(subset["higher_risk"])
    rd_lo, rd_hi = percentile_ci(subset["risk_difference"])

    summary_rows.append({
        **point_row.to_dict(),
        "bootstrap_B_requested": RISK_BOOTSTRAP_B,
        "bootstrap_B_successful": int(subset["replicate"].nunique()),
        "lower_risk_CI_low": low_lo,
        "lower_risk_CI_high": low_hi,
        "higher_risk_CI_low": high_lo,
        "higher_risk_CI_high": high_hi,
        "risk_difference_CI_low": rd_lo,
        "risk_difference_CI_high": rd_hi,
    })

risk_summary = pd.DataFrame(summary_rows)
risk_summary["N"] = len(risk_use)
risk_summary["events"] = int(risk_use.event.sum())
risk_summary["time_days"] = risk_summary["time"].map(RISK_TIMES_DAYS)

risk_summary.to_csv(
    OUT / "10_adjusted_36_60_month_absolute_risk.csv",
    index=False,
)
risk_boot.to_csv(
    OUT / "10_adjusted_36_60_month_absolute_risk_bootstrap.csv",
    index=False,
)

display(
    risk_summary.assign(
        lower_adjusted_risk_percent=100 * risk_summary["lower_adjusted_risk"],
        higher_adjusted_risk_percent=100 * risk_summary["higher_adjusted_risk"],
        absolute_risk_difference_percent_points=100 * risk_summary["absolute_risk_difference"],
        risk_difference_CI_low_percent_points=100 * risk_summary["risk_difference_CI_low"],
        risk_difference_CI_high_percent_points=100 * risk_summary["risk_difference_CI_high"],
    )[
        [
            "contrast", "time",
            "lower_adjusted_risk_percent",
            "higher_adjusted_risk_percent",
            "absolute_risk_difference_percent_points",
            "risk_difference_CI_low_percent_points",
            "risk_difference_CI_high_percent_points",
            "bootstrap_B_successful",
        ]
    ].round(3)
)
if len(technical_table) != 7 or len(alternative_table) != 6 or len(motion_sensitivity_table) != 4:
    raise RuntimeError("A required technical model did not complete")
if not (risk_summary.bootstrap_B_successful == RISK_BOOTSTRAP_B).all():
    raise RuntimeError("Incomplete absolute-risk bootstrap; inspect replicate CSV")
