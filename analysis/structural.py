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

SENSITIVITY_MATCH_WINDOW_DAYS = 365
RUN_365D_SENSITIVITY = True
FS_PATH = DATA_DIR / "structural_mri_volumes.csv"

# ============================================================
# 2. Load cohort and construct primary variables
# ============================================================

core = pd.read_excel(COHORT_PATH, sheet_name=SHEET_NAME, engine="openpyxl")

required_core = [
    "PTID", "INDEX_DATE", "BASE_DX",
    "AGE_INDEX", "SEX(M = 0)", "EDUCATION", "BASE_MMSE",
    "EVENT_ADconversion", "TIME_BY_DAYS",
    "WMH", "BRAIN_PARENCHYMA_FRACTION",
    "alps_L", "alps_R", "ALPS",
]
missing = [c for c in required_core if c not in core.columns]
if missing:
    raise KeyError(f"Missing required cohort columns: {missing}")

def parse_adni_date(s):
    # Handle either Excel serial dates or ordinary date strings.
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(
            s, unit="D", origin="1899-12-30", errors="coerce"
        ).dt.normalize()
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd

core["INDEX_DATE_PARSED"] = parse_adni_date(core["INDEX_DATE"])
core["mean_alps_calc"] = (
    pd.to_numeric(core["alps_L"], errors="coerce") +
    pd.to_numeric(core["alps_R"], errors="coerce")
) / 2
core["absolute_ai_raw"] = (
    pd.to_numeric(core["alps_L"], errors="coerce") -
    pd.to_numeric(core["alps_R"], errors="coerce")
).abs()
core["log_wmh"] = np.log1p(
    pd.to_numeric(core["WMH"], errors="coerce")
)

full_scale_vars = [
    "AGE_INDEX", "EDUCATION", "BASE_MMSE",
    "BRAIN_PARENCHYMA_FRACTION", "log_wmh",
    "mean_alps_calc", "absolute_ai_raw",
]

scaling_rows = []
for c in full_scale_vars:
    v = pd.to_numeric(core[c], errors="coerce")
    scaling_rows.append({
        "variable": c,
        "mean_full_cohort": v.mean(),
        "sd_full_cohort_ddof0": v.std(ddof=0),
        "n": v.notna().sum(),
    })
    core[f"{c}_z"] = zscore(v)

scaling = pd.DataFrame(scaling_rows)
scaling.to_csv(TABLE_DIR / "00_full_cohort_scaling.csv", index=False)

mean_error = (
    pd.to_numeric(core["ALPS"], errors="coerce") - core["mean_alps_calc"]
).abs().max()

print(f"Cohort N = {len(core):,}")
print(f"AD conversion events = {int(core['EVENT_ADconversion'].sum()):,}")
print(f"Stored vs recalculated mean ALPS max abs error = {mean_error:.3e}")
display(scaling)

# ============================================================
# 3. Load and deduplicate FreeSurfer table
# ============================================================

fs = pd.read_csv(FS_PATH, low_memory=False)

required_fs = [
    "PTID", "EXAMDATE", "STATUS", "OVERALLQC",
    "ST37SV", "ST96SV",
    "ST147SV", "ST148SV",
    "ST150SV", "ST151SV",
    "ST10CV",
]
missing_fs = [c for c in required_fs if c not in fs.columns]
if missing_fs:
    raise KeyError(f"Missing required FreeSurfer columns: {missing_fs}")

fs["EXAMDATE_PARSED"] = pd.to_datetime(
    fs["EXAMDATE"], errors="coerce"
).dt.normalize()

struct_fs_cols = [
    "ST37SV", "ST96SV", "ST147SV", "ST148SV",
    "ST150SV", "ST151SV", "ST10CV",
]
if {"ST30SV", "ST89SV"}.issubset(fs.columns):
    struct_fs_cols += ["ST30SV", "ST89SV"]

for c in struct_fs_cols:
    fs[c] = pd.to_numeric(fs[c], errors="coerce")

fs["_status_rank"] = np.where(
    fs["STATUS"].astype(str).str.lower().eq("complete"), 0,
    np.where(fs["STATUS"].astype(str).str.lower().eq("partial"), 1, 2),
)
qc_order = {"Pass": 0, "Partial": 1, "Hippocampus Only": 2, "Fail": 3}
fs["_qc_rank"] = fs["OVERALLQC"].map(qc_order).fillna(9)
fs["_n_struct_nonmissing"] = fs[struct_fs_cols].notna().sum(axis=1)

fs_dedup = (
    fs.sort_values(
        ["PTID", "EXAMDATE_PARSED", "_status_rank", "_qc_rank", "_n_struct_nonmissing"],
        ascending=[True, True, True, True, False],
    )
    .drop_duplicates(["PTID", "EXAMDATE_PARSED"], keep="first")
    .copy()
)

print(f"Raw FreeSurfer rows: {len(fs):,}")
print(f"Unique PTID/date rows after deterministic selection: {len(fs_dedup):,}")

# ============================================================
# 4. Matching functions and QC
# ============================================================

def exact_match_core_fs(core_df, fs_df):
    out = core_df.merge(
        fs_df,
        left_on=["PTID", "INDEX_DATE_PARSED"],
        right_on=["PTID", "EXAMDATE_PARSED"],
        how="left",
        suffixes=("", "_FS"),
    )
    out["FS_GAP_DAYS"] = np.where(
        out["EXAMDATE_PARSED"].notna(),
        (out["EXAMDATE_PARSED"] - out["INDEX_DATE_PARSED"]).dt.days.abs(),
        np.nan,
    )
    return out

def nearest_match_core_fs(core_df, fs_df, max_days=365):
    by_ptid = {
        k: g.sort_values("EXAMDATE_PARSED")
        for k, g in fs_df.groupby("PTID")
    }
    fs_payload_cols = [c for c in fs_df.columns if c != "PTID"]
    rows = []

    for _, r in core_df.iterrows():
        base = r.to_dict()
        ptid = r["PTID"]
        idx_date = r["INDEX_DATE_PARSED"]
        g = by_ptid.get(ptid)

        if g is None or pd.isna(idx_date) or g.empty:
            for c in fs_payload_cols:
                base[c] = np.nan
            base["FS_GAP_DAYS"] = np.nan
            rows.append(base)
            continue

        gaps = (g["EXAMDATE_PARSED"] - idx_date).dt.days.abs()
        j = gaps.idxmin()
        gap = float(gaps.loc[j])

        if gap <= max_days:
            chosen = g.loc[j]
            for c in fs_payload_cols:
                base[c] = chosen[c]
        else:
            for c in fs_payload_cols:
                base[c] = np.nan

        base["FS_GAP_DAYS"] = gap
        rows.append(base)

    return pd.DataFrame(rows)

exact = exact_match_core_fs(core, fs_dedup)
same_day = exact.loc[exact["EXAMDATE_PARSED"].notna()].copy()

match_qc = pd.DataFrame([{
    "cohort_n": len(core),
    "cohort_events": int(core["EVENT_ADconversion"].sum()),
    "same_day_match_n": int(len(same_day)),
    "same_day_events": int(same_day["EVENT_ADconversion"].sum()),
    "same_day_pct": float(len(same_day) / len(core) * 100),
}])

match_qc.to_csv(TABLE_DIR / "01_match_QC.csv", index=False)
display(match_qc)

print("STATUS among same-day matches")
display(same_day["STATUS"].value_counts(dropna=False).rename("n").to_frame())

print("OVERALLQC among same-day matches")
display(same_day["OVERALLQC"].value_counts(dropna=False).rename("n").to_frame())

# ============================================================
# 5. Structural asymmetry construction
# ============================================================

def relative_abs_asymmetry(left, right):
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    denom = left + right
    out = 2 * (left - right).abs() / denom
    return out.where(denom > 0)

def add_structural_metrics(df):
    d = df.copy()

    d["vent_ai"] = relative_abs_asymmetry(d["ST37SV"], d["ST96SV"])
    d["gm_ai"] = relative_abs_asymmetry(d["ST147SV"], d["ST148SV"])
    d["wm_ai"] = relative_abs_asymmetry(d["ST150SV"], d["ST151SV"])

    icv = pd.to_numeric(d["ST10CV"], errors="coerce")
    vent_total = (
        pd.to_numeric(d["ST37SV"], errors="coerce") +
        pd.to_numeric(d["ST96SV"], errors="coerce")
    )
    d["vent_total_icv"] = (vent_total / icv).where(icv > 0)

    return d

same_day = add_structural_metrics(same_day)

struct_vars = ["vent_ai", "gm_ai", "wm_ai", "vent_total_icv"]

desc_vars = ["absolute_ai_raw"] + struct_vars
desc = same_day[desc_vars].describe(
    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
).T
desc["missing_n"] = same_day[desc_vars].isna().sum()
desc.to_csv(TABLE_DIR / "02_structural_asymmetry_descriptives.csv")

display(desc)

# ============================================================
# 6. Correlation analyses
# ============================================================

corr_targets = ["vent_ai", "gm_ai", "wm_ai"]
if "vent_ai_extended" in same_day.columns:
    corr_targets.append("vent_ai_extended")

corr_rows = []
for y in corr_targets:
    dd = same_day[["absolute_ai_raw", y]].dropna()
    pear_r, pear_p = stats.pearsonr(dd["absolute_ai_raw"], dd[y])
    spear_r, spear_p = stats.spearmanr(dd["absolute_ai_raw"], dd[y])

    corr_rows.append({
        "structural_metric": y,
        "n": len(dd),
        "pearson_r": pear_r,
        "pearson_p": pear_p,
        "spearman_rho": spear_r,
        "spearman_p": spear_p,
    })

corr_table = pd.DataFrame(corr_rows)
corr_table.to_csv(TABLE_DIR / "03_absALPS_structural_correlations.csv", index=False)
display(corr_table)

corr_matrix = same_day[
    ["absolute_ai_raw", "vent_ai", "gm_ai", "wm_ai"]
].corr(method="spearman")
corr_matrix.to_csv(TABLE_DIR / "03b_spearman_matrix.csv")
display(corr_matrix)

# ============================================================
# 7. Cox helper functions + full-cohort reproduction
# ============================================================

BASE_TERMS = [
    "AGE_INDEX_z",
    "sex_code",
    "EDUCATION_z",
    "BASE_MMSE_z",
    "log_wmh_z",
    "BRAIN_PARENCHYMA_FRACTION_z",
    "mean_alps_calc_z",
]
ABS_AI_TERM = "absolute_ai_raw_z"
FULL_TERMS = BASE_TERMS + [ABS_AI_TERM]

def prepare_cox_columns(df):
    d = df.copy()
    d["sex_code"] = pd.to_numeric(d["SEX(M = 0)"], errors="coerce")
    return d

def fit_cox(df, terms, label, robust=True):
    needed = ["TIME_BY_DAYS", "EVENT_ADconversion", "BASE_DX"] + terms
    dd = df[needed].dropna().copy()

    cph = CoxPHFitter()
    used_penalizer = 0.0

    cph.fit(
        dd,
        duration_col="TIME_BY_DAYS",
        event_col="EVENT_ADconversion",
        strata=["BASE_DX"],
        formula=" + ".join(terms),
        robust=robust,
    )

    return cph, dd, used_penalizer

def extract_term(cph, term, model_label, n, events, penalizer=0.0):
    s = cph.summary.loc[term]
    return {
        "model": model_label,
        "n": int(n),
        "events": int(events),
        "term": term,
        "beta": float(s["coef"]),
        "HR": float(s["exp(coef)"]),
        "CI_low": float(s["exp(coef) lower 95%"]),
        "CI_high": float(s["exp(coef) upper 95%"]),
        "p": float(s["p"]),
        "log_likelihood": float(cph.log_likelihood_),
        "c_index": float(cph.concordance_index_),
        "penalizer": float(penalizer),
    }

core_cox = prepare_cox_columns(core)
primary_full, primary_full_df, primary_pen = fit_cox(
    core_cox,
    FULL_TERMS,
    "Original full-cohort primary model",
)

primary_reproduction = pd.DataFrame([
    extract_term(
        primary_full,
        ABS_AI_TERM,
        "Original full-cohort primary model",
        len(primary_full_df),
        primary_full_df["EVENT_ADconversion"].sum(),
        primary_pen,
    )
])

primary_reproduction.to_csv(
    TABLE_DIR / "04_primary_Cox_reproduction.csv", index=False
)

display(primary_reproduction)
display(
    primary_full.summary[
        ["coef", "exp(coef)", "exp(coef) lower 95%",
         "exp(coef) upper 95%", "p"]
    ]
)

# ============================================================
# 8. Structural-asymmetry model ladder
# ============================================================

sd = prepare_cox_columns(same_day)

common_required = [
    "TIME_BY_DAYS", "EVENT_ADconversion", "BASE_DX",
    *FULL_TERMS,
    "vent_total_icv", "vent_ai", "gm_ai", "wm_ai",
]
analysis_df = sd.dropna(subset=common_required).copy()

for c in ["vent_total_icv", "vent_ai", "gm_ai", "wm_ai"]:
    analysis_df[f"{c}_z"] = zscore(analysis_df[c])


print(f"Common same-day analysis N = {len(analysis_df):,}")
print(f"Events = {int(analysis_df['EVENT_ADconversion'].sum()):,}")

model_specs = {
    "S0_reference_same_sample": FULL_TERMS,
    "S3_plus_vent_burden_and_asymmetry": FULL_TERMS + [
        "vent_total_icv_z", "vent_ai_z"
    ],
    "S5_final_all_structural_asymmetry": FULL_TERMS + [
        "vent_total_icv_z", "vent_ai_z", "gm_ai_z", "wm_ai_z"
    ],
}


models = {}
model_rows = []
penalizers = {}

for name, terms in model_specs.items():
    cph, dd, pen = fit_cox(analysis_df, terms, name)
    models[name] = cph
    penalizers[name] = pen

    row = extract_term(
        cph,
        ABS_AI_TERM,
        name,
        len(dd),
        dd["EVENT_ADconversion"].sum(),
        pen,
    )
    row["structural_terms"] = ", ".join(
        t for t in terms
        if t in {
            "vent_total_icv_z", "vent_ai_z", "gm_ai_z", "wm_ai_z",
        }
    )
    model_rows.append(row)

cox_ladder = pd.DataFrame(model_rows)

ref_beta = cox_ladder.loc[
    cox_ladder["model"].eq("S0_reference_same_sample"), "beta"
].iloc[0]

cox_ladder["AbsoluteAI_beta_attenuation_pct_vs_S0"] = (
    100 * (1 - cox_ladder["beta"] / ref_beta)
)

cox_ladder.to_csv(
    TABLE_DIR / "05_structural_asymmetry_Cox_ladder.csv", index=False
)
display(cox_ladder)

# ============================================================
# 11. ±365-day sensitivity
# ============================================================

sensitivity_365_result = pd.DataFrame()

if RUN_365D_SENSITIVITY:
    near365 = nearest_match_core_fs(
        core,
        fs_dedup,
        max_days=SENSITIVITY_MATCH_WINDOW_DAYS,
    )
    near365 = add_structural_metrics(near365)
    near365 = prepare_cox_columns(near365)

    common_365 = [
        "TIME_BY_DAYS", "EVENT_ADconversion", "BASE_DX",
        *FULL_TERMS,
        "vent_total_icv", "vent_ai", "gm_ai", "wm_ai",
    ]
    d365 = near365.dropna(subset=common_365).copy()

    for c in ["vent_total_icv", "vent_ai", "gm_ai", "wm_ai"]:
        d365[f"{c}_z"] = zscore(d365[c])

    terms365 = FULL_TERMS + [
        "vent_total_icv_z", "vent_ai_z", "gm_ai_z", "wm_ai_z"
    ]

    m365, m365_df, pen365 = fit_cox(
        d365,
        terms365,
        "365-day final sensitivity",
    )

    sensitivity_365_result = pd.DataFrame([
        extract_term(
            m365,
            ABS_AI_TERM,
            "365-day final sensitivity",
            len(m365_df),
            m365_df["EVENT_ADconversion"].sum(),
            pen365,
        )
    ])
    sensitivity_365_result["max_allowed_gap_days"] = (
        SENSITIVITY_MATCH_WINDOW_DAYS
    )
    sensitivity_365_result.to_csv(
        TABLE_DIR / "10_365day_structural_sensitivity.csv",
        index=False,
    )

    print(f"365-day common N = {len(d365):,}")
    print(f"Events = {int(d365['EVENT_ADconversion'].sum()):,}")
    display(sensitivity_365_result)
else:
    print("365-day sensitivity skipped.")
