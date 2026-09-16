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


from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from importlib.metadata import version
import hashlib, json, os, platform, random, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import torch
import torchtuples as tt
from pycox.models import CoxPH, CoxTime, DeepHitSingle
from pycox.models.cox_time import MLPVanillaCoxTime

@dataclass(frozen=True)
class Config:
    seed: int = 20260828
    data_stem: str = "baseline_clinical_alps"
    analysis_version: str = "absolute_only_v3"
    horizons: Tuple[float, ...] = (12.0, 36.0, 60.0)
    ibs_grid: Tuple[float, ...] = tuple(np.linspace(12.0, 60.0, 17))
    outer_folds: int = 5
    core_repeats: int = 5
    training_seeds: Tuple[int, ...] = (111, 222, 333)
    max_epochs: int = 256
    patience: int = 25
    batch_size: int = 256
    hidden_nodes: Tuple[int, ...] = (16, 8)
    dropout: float = 0.15
    learning_rate: float = 1e-3
    deephit_bins: int = 24
    deephit_alpha: float = 0.2
    deephit_sigma: float = 0.1
    bootstrap_n: int = SETTINGS["prediction_bootstrap"]
    min_horizon_cases: int = 20
    min_horizon_controls: int = 20
    min_censor_survival: float = 0.05
    min_boot_valid_fraction: float = 0.9
    run_subgroup_evaluation: bool = False
    device: str = SETTINGS["device"]

CFG = Config()
assert CFG.bootstrap_n >= 2 and CFG.outer_folds >= 2 and CFG.core_repeats >= 1
assert CFG.training_seeds and 0 < CFG.min_censor_survival < 1
DAY_PER_MONTH = 365.25 / 12.0
PRED_TIMES = np.unique(np.r_[CFG.ibs_grid, CFG.horizons])
MODEL_FAMILIES = ("DeepSurv", "CoxTime", "DeepHitSingle")
REFERENCE = ["age", "sex_female", "education", "mmse", "diagnosis_mci", "bpf", "log_wmh", "mean_alps"]
FEATURE_BLOCKS = {"reference": REFERENCE, "extended": REFERENCE + ["absolute_ai"]}

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PACKAGE_VERSIONS = {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "torchtuples", "pycox"]}
seed_everything(CFG.seed)
print({"device": CFG.device, "packages": PACKAGE_VERSIONS})

COLUMN_ALIASES = {
    "id": ["PTID", "RID", "subject_id", "ID"],
    "diagnosis": ["BASE_DX", "baseline_diagnosis", "DX"],
    "time_days": ["TIME_BY_DAYS", "time_days", "followup_days"],
    "time_months": ["TIME_MONTHS", "time_months", "followup_months"],
    "event": ["EVENT_ADconversion", "event", "AD_conversion"],
    "age": ["AGE_INDEX", "AGE", "age"],
    "sex": ["SEX(M = 0)", "SEX", "sex", "female"],
    "education": ["EDUCATION", "PTEDUCAT", "education"],
    "mmse": ["BASE_MMSE", "MMSE", "baseline_mmse"],
    "wmh": ["FS_WMH_ML", "WMH", "WMH_VOLUME", "wmh"],
    "bpf": ["FS_BPF", "BRAIN_PARENCHYMA_FRACTION", "BPF", "bpf"],
    "alps": ["ALPS", "mean_ALPS", "aDTI_ALPS", "alps"],
    "alps_l": ["alps_L", "ALPS_L", "left_ALPS", "ALPS_left"],
    "alps_r": ["alps_R", "ALPS_R", "right_ALPS", "ALPS_right"],
    "x_proj_l": ["x_proj_L", "X_PROJ_L"], "x_assoc_l": ["x_assoc_L", "X_ASSOC_L"],
    "y_proj_l": ["y_proj_L", "Y_PROJ_L"], "z_assoc_l": ["z_assoc_L", "Z_ASSOC_L"],
    "x_proj_r": ["x_proj_R", "X_PROJ_R"], "x_assoc_r": ["x_assoc_R", "X_ASSOC_R"],
    "y_proj_r": ["y_proj_R", "Y_PROJ_R"], "z_assoc_r": ["z_assoc_R", "Z_ASSOC_R"],
}

TENSOR_KEYS = ["x_proj_l", "x_assoc_l", "y_proj_l", "z_assoc_l", "x_proj_r", "x_assoc_r", "y_proj_r", "z_assoc_r"]
CORE_KEYS = ["id", "diagnosis", "event", "age", "sex", "education", "mmse", "wmh", "bpf", "alps"]

def resolve_columns(columns: Sequence[str]) -> Dict[str, str]:
    lookup = {str(c).strip().lower(): c for c in columns}
    out: Dict[str, str] = {}
    for key, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.strip().lower() in lookup:
                out[key] = lookup[alias.strip().lower()]
                break
    missing = [k for k in CORE_KEYS if k not in out]
    if "time_days" not in out and "time_months" not in out:
        missing.append("time_days or time_months")
    if not ({"alps_l", "alps_r"} <= set(out)) and not all(k in out for k in TENSOR_KEYS):
        missing.append("bilateral ALPS or all eight tensor columns")
    if missing:
        raise KeyError("Missing variables: " + ", ".join(missing) + "\nAvailable:\n" + "\n".join(map(str, columns)))
    return out

def encode_diagnosis(s: pd.Series) -> pd.Series:
    text = s.astype(str).str.strip().str.upper()
    def one(x: str):
        if "MCI" in x:
            return "MCI"
        if x in {"CN", "NC", "NORMAL"} or "COGNITIVELY NORMAL" in x:
            return "CN"
        return np.nan
    return text.map(one)

def encode_sex(s):
    num = pd.to_numeric(s, errors="coerce")
    text = s.astype("string").str.strip().str.upper()
    mapped = text.map({"F": 1., "FEMALE": 1., "여": 1., "M": 0., "MALE": 0., "남": 0.})
    return num.where(num.isin([0, 1]), mapped).astype(float)


def build_frame(source, col):
    d = pd.DataFrame(index=source.index)
    d["subject_id"] = source[col["id"]].astype("string").str.strip()
    d["diagnosis"] = encode_diagnosis(source[col["diagnosis"]])
    d["diagnosis_mci"] = d.diagnosis.eq("MCI").astype(float)
    def num(key): return pd.to_numeric(source[col[key]], errors="coerce")
    d["event"] = num("event")
    d["time_months"] = num("time_days") / DAY_PER_MONTH
    for key in ["age", "education", "mmse", "wmh", "bpf"]: d[key] = num(key)
    d["sex_female"] = encode_sex(source[col["sex"]])
    d["log_wmh"] = np.log1p(d.wmh.where(d.wmh.ge(0)))
    d["mean_alps"] = num("alps")
    has_lr = {"alps_l", "alps_r"} <= set(col)
    if has_lr:
        d["alps_l"], d["alps_r"] = num("alps_l"), num("alps_r")
    else:
        # Tensor columns are used only to reconstruct the two required ALPS values.
        for side in ["l", "r"]:
            numerator = (num(f"x_proj_{side}") + num(f"x_assoc_{side}")) / 2
            denominator = (num(f"y_proj_{side}") + num(f"z_assoc_{side}")) / 2
            d[f"alps_{side}"] = numerator.where(numerator.gt(0)) / denominator.where(denominator.gt(0))
    d["absolute_ai"] = (d.alps_l - d.alps_r).abs()
    required = list(dict.fromkeys(REFERENCE + ["alps_l", "alps_r", "absolute_ai", "event", "time_months"]))
    bad = pd.DataFrame(index=d.index)
    bad["invalid_id"] = d.subject_id.isna() | d.subject_id.str.lower().isin(["", "nan", "none", "null", "<na>"])
    bad["invalid_diagnosis"] = d.diagnosis.isna()
    bad["missing_or_nonfinite"] = ~np.isfinite(d[required].to_numpy(float)).all(axis=1)
    bad["invalid_outcome"] = ~d.event.isin([0, 1]) | ~d.time_months.gt(0)
    bad["invalid_alps"] = ~d[["mean_alps", "alps_l", "alps_r"]].gt(0).all(axis=1)
    bad["negative_wmh"] = d.wmh.lt(0)
    reject = bad.any(axis=1)
    exclusions = pd.DataFrame({"source_row": np.arange(len(d)) + 2,
        "subject_id": d.subject_id, "reason": bad.apply(lambda r: "|".join(r.index[r]), axis=1)})
    exclusions = exclusions.loc[reject].reset_index(drop=True)
    d = d.loc[~reject].copy().sort_values("subject_id").reset_index(drop=True)
    if d.empty: raise ValueError("No eligible complete-case rows")
    if d.subject_id.duplicated().any(): raise ValueError("Duplicate subject IDs; select one baseline row per person")
    d["subject_id"] = d.subject_id.astype(str)
    d["event"] = d.event.astype(int)
    np.testing.assert_allclose(d.absolute_ai, (d.alps_l-d.alps_r).abs(), rtol=0, atol=1e-12)
    mean_error = (d.mean_alps - (d.alps_l+d.alps_r)/2).abs()
    qc = pd.DataFrame({"metric": ["source_n", "complete_case_n", "excluded_n", "events",
        "stored_vs_LR_mean_max_abs_error", "absolute_ai_mean", "absolute_ai_sd"],
        "value": [len(source), len(d), int(reject.sum()), int(d.event.sum()), mean_error.max(), d.absolute_ai.mean(), d.absolute_ai.std()]})
    if mean_error.max() > 1e-4:
        warnings.warn("Stored mean differs from (L+R)/2. Review QC; the stored study mean is retained.")
    return d, qc, exclusions

DATA_PATH = DATA_DIR / "baseline_clinical_alps.xlsx"
SOURCE = pd.read_excel(DATA_PATH, sheet_name="survival_dataset_final_age")
COLMAP = resolve_columns(SOURCE.columns)
df, qc_table, exclusions = build_frame(SOURCE, COLMAP)
display(qc_table)

def inner_train_val_indices(train_df, seed):
    strata = train_df.diagnosis.astype(str) + "_E" + train_df.event.astype(int).astype(str)
    if strata.value_counts().min() < 5:
        strata = train_df.event.astype(int).astype(str)
    if strata.value_counts().min() < 5:
        raise ValueError("Too few events/non-events for the five-way inner split")
    tr, va = next(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(train_df)), strata))
    if train_df.iloc[tr].event.sum() == 0 or train_df.iloc[va].event.sum() == 0:
        raise ValueError("Training or validation set has no events")
    return tr, va


def preprocess_fit(train_df, val_df, test_df, features):
    scaler = StandardScaler().fit(train_df[features])
    arrays = [scaler.transform(d[features]).astype("float32") for d in [train_df, val_df, test_df]]
    if not all(np.isfinite(a).all() for a in arrays): raise ValueError("Nonfinite model input")
    return (*arrays, scaler)


def interpolate_survival(surv_df, query_times):
    times = np.asarray(surv_df.index, float)
    values = np.asarray(surv_df.values, float).T
    order = np.argsort(times)
    times, values = times[order], values[:, order]
    if len(times) == 0 or not np.isfinite(values).all(): raise ValueError("Invalid predicted survival curve")
    if np.any(np.diff(values, axis=1) > 1e-5): raise ValueError("Non-monotone survival curve")
    out = np.vstack([np.interp(query_times, times, row, left=1., right=np.nan) for row in values])
    return np.clip(out, 0., 1.)


def fit_one_model(model_name: str, x_train: np.ndarray, y_train: Tuple[np.ndarray, np.ndarray],
                  x_val: np.ndarray, y_val: Tuple[np.ndarray, np.ndarray], x_test: np.ndarray,
                  seed: int) -> Tuple[np.ndarray, Dict[str, object]]:
    seed_everything(seed)
    durations_train, events_train = y_train
    durations_val, events_val = y_val
    in_features = x_train.shape[1]
    callbacks = [tt.callbacks.EarlyStopping(patience=CFG.patience)]
    common_net = dict(num_nodes=list(CFG.hidden_nodes), batch_norm=False, dropout=CFG.dropout)
    if model_name == "DeepSurv":
        net = tt.practical.MLPVanilla(in_features, common_net["num_nodes"], 1,
                                      common_net["batch_norm"], common_net["dropout"], output_bias=False)
        model = CoxPH(net, tt.optim.Adam, device=CFG.device)
        model.optimizer.set_lr(CFG.learning_rate)
        # Full risk-set batch for the small cohort; avoids minibatch Cox partial-likelihood drift.
        log = model.fit(x_train, (durations_train, events_train), batch_size=len(x_train),
                        epochs=CFG.max_epochs, callbacks=callbacks, verbose=False,
                        val_data=(x_val, (durations_val, events_val)))
        model.compute_baseline_hazards()
        surv_df = model.predict_surv_df(x_test)
    elif model_name == "CoxTime":
        labtrans = CoxTime.label_transform()
        yt = labtrans.fit_transform(durations_train, events_train)
        yv = labtrans.transform(durations_val, events_val)
        net = MLPVanillaCoxTime(in_features, common_net["num_nodes"], common_net["batch_norm"], common_net["dropout"])
        model = CoxTime(net, tt.optim.Adam, labtrans=labtrans, device=CFG.device)
        model.optimizer.set_lr(CFG.learning_rate)
        log = model.fit(x_train, yt, batch_size=CFG.batch_size, epochs=CFG.max_epochs,
                        callbacks=callbacks, verbose=False, val_data=(x_val, yv))
        model.compute_baseline_hazards()
        surv_df = model.predict_surv_df(x_test)
    elif model_name == "DeepHitSingle":
        labtrans = DeepHitSingle.label_transform(CFG.deephit_bins)
        yt = labtrans.fit_transform(durations_train, events_train)
        yv = labtrans.transform(durations_val, events_val)
        out_features = labtrans.out_features
        net = tt.practical.MLPVanilla(in_features, common_net["num_nodes"], out_features,
                                      common_net["batch_norm"], common_net["dropout"])
        model = DeepHitSingle(net, tt.optim.Adam, alpha=CFG.deephit_alpha, sigma=CFG.deephit_sigma,
                              duration_index=labtrans.cuts, device=CFG.device)
        model.optimizer.set_lr(CFG.learning_rate)
        log = model.fit(x_train, yt, batch_size=CFG.batch_size, epochs=CFG.max_epochs,
                        callbacks=callbacks, verbose=False, val_data=(x_val, yv))
        surv_df = model.interpolate(10).predict_surv_df(x_test)
    else:
        raise ValueError(model_name)
    pred = interpolate_survival(surv_df, PRED_TIMES)
    history = log.to_pandas()
    val_cols = history.filter(like="val")
    val_series = val_cols.iloc[:, 0] if len(val_cols.columns) else pd.Series(dtype=float)
    meta = {"stopped_epoch": int(history.index[-1]) if len(history) else None,
            "best_epoch": int(val_series.idxmin()) if val_series.notna().any() else None,
            "min_val_loss": float(val_series.min()) if val_series.notna().any() else None,
            "history": history.reset_index().to_dict(orient="records")}
    return pred, meta


def censor_km(train_time, train_event):
    t = np.asarray(train_time, float)
    e = np.asarray(train_event, int)
    times, inv, counts = np.unique(t, return_inverse=True, return_counts=True)
    censored = np.bincount(inv, weights=(1-e), minlength=len(times))
    at_risk = np.cumsum(counts[::-1])[::-1]
    # Conventional reverse-event KM; events and censoring at the same recorded time share the risk set.
    survival = np.cumprod(1-censored/at_risk)
    return times, survival


def km_lookup(km, queries, left=False):
    times, survival = km
    q = np.asarray(queries, float)
    idx = np.searchsorted(times, q, side="left" if left else "right") - 1
    return np.where(idx < 0, 1., survival[np.maximum(idx, 0)])


def eval_weight_columns(test_df, train_df, horizons):
    km = censor_km(train_df.time_months, train_df.event)
    t, e = test_df.time_months.to_numpy(float), test_df.event.to_numpy(int)
    ge = km_lookup(km, t, left=True)
    out = {}
    for h in horizons:
        tag = f"{h:g}"
        case, control = (e == 1) & (t <= h), t > h
        gh = float(km_lookup(km, h))
        supported = h < train_df.time_months.max() and gh >= CFG.min_censor_survival
        supported = supported and bool(np.all(ge[case] >= CFG.min_censor_survival))
        wc, wn = np.zeros(len(t)), np.zeros(len(t))
        if supported:
            wc[case] = 1/ge[case]
            wn[control] = 1/gh
        out.update({f"case_{tag}": case.astype(int), f"control_{tag}": control.astype(int),
                    f"wc_{tag}": wc, f"wn_{tag}": wn, f"supported_{tag}": np.full(len(t), int(supported)),
                    f"G_{tag}": np.full(len(t), gh)})
    return pd.DataFrame(out)

def dataframe_hash(d):
    stable = d.sort_values("subject_id").reset_index(drop=True)
    return hashlib.sha256(stable.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path, d, compression=None):
    tmp = path.with_name(path.name + ".tmp")
    d.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)




def make_master_splits(d):
    strata = d.diagnosis.astype(str) + "_E" + d.event.astype(int).astype(str)
    if strata.value_counts().min() < CFG.outer_folds:
        warnings.warn("Sparse diagnosis/event stratum: outer split stratifies on event only")
        strata = d.event.astype(str)
    if strata.value_counts().min() < CFG.outer_folds: raise ValueError("Insufficient events/non-events for outer folds")
    rows = []
    for repeat in range(CFG.core_repeats):
        skf = StratifiedKFold(CFG.outer_folds, shuffle=True, random_state=CFG.seed+1009*repeat)
        for fold, (_, te) in enumerate(skf.split(np.zeros(len(d)), strata)):
            rows.extend({"subject_id": d.iloc[i].subject_id, "repeat": repeat, "outer_fold": fold} for i in te)
    reg = pd.DataFrame(rows).sort_values(["repeat", "subject_id"]).reset_index(drop=True)
    path = CHECKPOINT_DIR / "master_splits.csv"
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path, dtype={"subject_id":str}), reg, check_dtype=False)
    else: atomic_csv(path, reg)
    return reg




def validate_oof(out, model, block, expected):
    keys = ["repeat", "subject_id"]
    a = out.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    if a.duplicated(keys).any(): raise ValueError("Duplicate OOF keys")
    pd.testing.assert_frame_equal(a[keys+["outer_fold"]], b[keys+["outer_fold"]], check_dtype=False)
    if not a.model.eq(model).all() or not a.block.eq(block).all(): raise ValueError("Wrong OOF model/block")
    if not a.training_signature.eq(TRAINING_SIGNATURE).all(): raise ValueError("Stale OOF signature")
    truth = df.set_index("subject_id").loc[a.subject_id]
    np.testing.assert_allclose(a.time_months, truth.time_months, rtol=1e-12)
    np.testing.assert_array_equal(a.event, truth.event)
    np.testing.assert_array_equal(a.diagnosis, truth.diagnosis)
    for h in PRED_TIMES:
        tag = f"{h:g}"
        t, e = a.time_months.to_numpy(), a.event.to_numpy()
        np.testing.assert_array_equal(a[f"case_{tag}"], ((e==1)&(t<=h)).astype(int))
        np.testing.assert_array_equal(a[f"control_{tag}"], (t>h).astype(int))
        p = a[f"surv_{tag}"].to_numpy(float)
        if np.isinf(p).any() or ((p[np.isfinite(p)]<0)|(p[np.isfinite(p)]>1)).any(): raise ValueError("Invalid OOF survival")
        if not a[f"supported_{tag}"].isin([0,1]).all(): raise ValueError("Invalid support indicator")
        for c in [f"wc_{tag}",f"wn_{tag}",f"G_{tag}"]:
            if not np.isfinite(a[c]).all() or (a[c]<0).any(): raise ValueError("Invalid IPCW data")
    survival = a[[f"surv_{h:g}" for h in PRED_TIMES]].to_numpy(float)
    if np.any(np.diff(survival,axis=1)>1e-5): raise ValueError("Non-monotone OOF survival")
    return a




def run_block_model(model_name, block):
    parts = []
    for (repeat, fold), (expected, train, val, test, weights) in FOLDS.items():
        stem = f"oof__{model_name}__{block}__r{repeat:02d}__f{fold:02d}"
        path = CHECKPOINT_DIR / (stem + ".csv.gz")
        meta_path = CHECKPOINT_DIR / (stem + ".json")
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["training_signature"] != TRAINING_SIGNATURE: raise ValueError("Checkpoint metadata mismatch")
            part = validate_oof(pd.read_csv(path, dtype={"subject_id":str}), model_name, block, expected)
            if meta["features"] != FEATURE_BLOCKS[block]:
                raise ValueError("Saved feature set mismatch")
            if meta["train_ids"] != train.subject_id.tolist() or meta["validation_ids"] != val.subject_id.tolist():
                raise ValueError("Saved inner split mismatch")
            _, _, _, check_scaler = preprocess_fit(train, val, test, FEATURE_BLOCKS[block])
            np.testing.assert_allclose(meta["scaler_mean"], check_scaler.mean_, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(meta["scaler_scale"], check_scaler.scale_, rtol=1e-12, atol=1e-12)
            # Recomputed weights must also match the checkpoint.
            for c in weights:
                np.testing.assert_allclose(part[c], weights[c], rtol=1e-10, atol=1e-12)
        else:
            xtr, xva, xte, scaler = preprocess_fit(train, val, test, FEATURE_BLOCKS[block])
            preds, logs = [], []
            for train_seed in CFG.training_seeds:
                seed = CFG.seed+train_seed+10000*repeat+fold
                p, log = fit_one_model(model_name, xtr,
                    (train.time_months.to_numpy("float32"), train.event.to_numpy("int64")),
                    xva, (val.time_months.to_numpy("float32"), val.event.to_numpy("int64")), xte, seed)
                preds.append(p); logs.append({"seed":seed, **log})
            survival = np.mean(preds, axis=0)
            part = test[["subject_id","diagnosis","time_months","event"]].copy()
            part["repeat"], part["outer_fold"] = repeat, fold
            part["model"], part["block"] = model_name, block
            part["training_signature"] = TRAINING_SIGNATURE
            for j,h in enumerate(PRED_TIMES): part[f"surv_{h:g}"] = survival[:,j]
            part = pd.concat([part, weights], axis=1)
            part = validate_oof(part, model_name, block, expected)
            atomic_csv(path, part, compression="gzip")
            atomic_json(meta_path, {"training_signature":TRAINING_SIGNATURE, "seed_logs":logs,
                "features":FEATURE_BLOCKS[block], "scaler_mean":scaler.mean_.tolist(), "scaler_scale":scaler.scale_.tolist(),
                "train_ids":train.subject_id.tolist(), "validation_ids":val.subject_id.tolist()})
        parts.append(part)
    return validate_oof(pd.concat(parts, ignore_index=True), model_name, block, split_registry)

training_config = asdict(CFG)
for key in ["bootstrap_n", "min_horizon_cases", "min_horizon_controls", "min_boot_valid_fraction", "run_subgroup_evaluation"]:
    training_config.pop(key)
training_manifest = {"data_hash": dataframe_hash(df), "config": training_config,
    "features": FEATURE_BLOCKS, "code_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "packages": PACKAGE_VERSIONS,
    "python": platform.python_version()}
TRAINING_SIGNATURE = hashlib.sha256(json.dumps(training_manifest, sort_keys=True, default=str).encode()).hexdigest()
OUT_DIR = SECTION_OUT
TABLE_DIR, FIGURE_DIR, CHECKPOINT_DIR = [OUT_DIR / s for s in ["tables", "figures", "checkpoints"]]
for folder in [TABLE_DIR, FIGURE_DIR, CHECKPOINT_DIR]: folder.mkdir(parents=True, exist_ok=True)
# Saved OOF evaluation is read-only; training checkpoints use the source hash.
def validate_replay_manifest(stored, current):
    """Check data and model identity independently of the input filename label."""
    for key in ["data_hash", "config", "features"]:
        old_value, new_value = stored[key], current[key]
        if key == "config":
            # data_stem describes the input file; it is not a training parameter.
            # Keep the stored manifest intact for its original checkpoint signature.
            old_value = {k: v for k, v in old_value.items() if k != "data_stem"}
            new_value = {k: v for k, v in new_value.items() if k != "data_stem"}
        if json.dumps(old_value, sort_keys=True) != json.dumps(new_value, sort_keys=True):
            raise ValueError("Saved OOF input/config mismatch: " + key)


ORIGINAL_RUN = SETTINGS.get("prediction_cache")
if ORIGINAL_RUN:
    original = Path(ORIGINAL_RUN)
    stored = json.loads((original/"training_manifest.json").read_text(encoding="utf-8"))
    validate_replay_manifest(stored, training_manifest)
    TRAINING_SIGNATURE = hashlib.sha256(json.dumps(stored,sort_keys=True,default=str).encode()).hexdigest()
    training_manifest["replayed_training_signature"] = TRAINING_SIGNATURE
    training_manifest["original_packages"] = stored["packages"]
    CHECKPOINT_DIR = original/"checkpoints"
    if not (CHECKPOINT_DIR/"master_splits.csv").is_file():
        raise FileNotFoundError("Missing saved master splits")
    for model in MODEL_FAMILIES:
        for block in FEATURE_BLOCKS:
            for r in range(CFG.core_repeats):
                for f in range(CFG.outer_folds):
                    stem=f"oof__{model}__{block}__r{r:02d}__f{f:02d}"
                    for ext in [".json", ".csv.gz"]:
                        if not (CHECKPOINT_DIR/(stem+ext)).is_file():
                            raise FileNotFoundError("Incomplete saved OOF run: "+stem+ext)
    def fit_one_model(*args, **kwargs):
        raise RuntimeError("Saved OOF replay cannot retrain a missing fold")

atomic_json(OUT_DIR / "training_manifest.json", training_manifest)
atomic_json(OUT_DIR / "config.json", asdict(CFG))
qc_table.to_csv(TABLE_DIR / "01_data_QC.csv", index=False)
exclusions.to_csv(TABLE_DIR / "01_exclusions.csv", index=False)
pd.DataFrame(COLMAP.items(), columns=["analysis_name", "source_column"]).to_csv(TABLE_DIR / "01_column_map.csv", index=False)
pd.DataFrame([{"block": b, "features": "|".join(f)} for b, f in FEATURE_BLOCKS.items()]).to_csv(TABLE_DIR / "02_features.csv", index=False)


split_registry = make_master_splits(df)

# Cache fold-local preprocessing-independent censoring weights once, shared by all six model/block pairs.
FOLDS = {}
for repeat in range(CFG.core_repeats):
    for fold in range(CFG.outer_folds):
        expected = split_registry.query("repeat == @repeat and outer_fold == @fold")
        ids = set(expected.subject_id)
        test = df[df.subject_id.isin(ids)].copy().reset_index(drop=True)
        outer = df[~df.subject_id.isin(ids)].copy().reset_index(drop=True)
        tr, va = inner_train_val_indices(outer, CFG.seed+10000*repeat+fold)
        FOLDS[(repeat, fold)] = (expected, outer.iloc[tr], outer.iloc[va], test, eval_weight_columns(test, outer, PRED_TIMES))

print("Planned seed fits:", len(MODEL_FAMILIES)*len(FEATURE_BLOCKS)*CFG.core_repeats*CFG.outer_folds*len(CFG.training_seeds))
prediction_sets = {}
for model in MODEL_FAMILIES:
    for block in FEATURE_BLOCKS:
        print(f"Running {model} / {block}")
        # Fail fast; never produce a complete-looking report from partial model results.
        prediction_sets[(model,block)] = run_block_model(model,block)
print("All OOF folds validated:", OUT_DIR)

CHECKPOINT_DIR = SECTION_OUT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

def make_cube(pred):
    pred = pred.sort_values(["repeat","subject_id"])
    shape = (CFG.core_repeats,len(df),len(PRED_TIMES))
    def stack(prefix):
        return np.stack([pred[f"{prefix}_{h:g}"].to_numpy(float).reshape(shape[:2]) for h in PRED_TIMES],axis=2)
    return {k:stack(k) for k in ["surv","case","control","wc","wn","supported","G"]}


def prepare_metrics(cube, mask):
    idx = np.flatnonzero(mask)
    d = {k:v[:,idx,:] for k,v in cube.items()}
    d["idx"] = idx
    d["valid"] = np.all((d["supported"]==1)&np.isfinite(d["surv"]),axis=1)
    d["loss"] = (d["wc"]+d["wn"])*(d["control"]-d["surv"])**2
    d["auc_order"] = {}
    for h in CFG.horizons:
        j = int(np.flatnonzero(PRED_TIMES==h)[0])
        for r in range(CFG.core_repeats):
            risk = 1-d["surv"][r,:,j]
            order = np.argsort(risk,kind="mergesort")
            starts = np.r_[0,np.flatnonzero(np.diff(risk[order])!=0)+1] if len(idx) else np.array([],int)
            d["auc_order"][(r,j)] = (order, starts)
    return d


def finite_repeat_mean(x):
    x = np.asarray(x,float)
    # Do not silently drop unsupported repeats.
    return float(x.mean()) if len(x) and np.isfinite(x).all() else np.nan


def metric_vector(prepared, counts):
    d = prepared
    count = np.asarray(counts,float)[d["idx"]]
    total = count.sum()
    result = {}
    if total<=0:
        return {key:np.nan for key in METRIC_KEYS}
    bs = np.sum(d["loss"]*count[None,:,None],axis=1)/total
    bs[~d["valid"]] = np.nan
    for h in CFG.horizons:
        j = int(np.flatnonzero(PRED_TIMES==h)[0])
        auc = []
        for r in range(CFG.core_repeats):
            if not d["valid"][r,j]: auc.append(np.nan); continue
            order, starts = d["auc_order"][(r,j)]
            cw = (d["wc"][r,:,j]*count)[order]
            nw = (d["wn"][r,:,j]*count)[order]
            den = cw.sum()*nw.sum()
            if den<=0: auc.append(np.nan); continue
            cg, ng = np.add.reduceat(cw,starts), np.add.reduceat(nw,starts)
            auc.append(float(np.sum(cg*(np.cumsum(ng)-ng+0.5*ng))/den))
        result[(float(h),"AUC")] = finite_repeat_mean(auc)
        result[(float(h),"Brier")] = finite_repeat_mean(bs[:,j])
    jj = [int(np.flatnonzero(PRED_TIMES==h)[0]) for h in CFG.ibs_grid]
    result[("12_60","IBS")] = finite_repeat_mean(trapezoid(bs[:,jj],x=CFG.ibs_grid,axis=1)/(max(CFG.ibs_grid)-min(CFG.ibs_grid)))
    return result

METRIC_KEYS = [(float(h),m) for h in CFG.horizons for m in ["AUC","Brier"]]+[("12_60","IBS")]
GROUP_MASKS = {"All":np.ones(len(df),bool)}
CUBES = {key:make_cube(pred) for key,pred in prediction_sets.items()}
PREPARED = {(model,block,group):prepare_metrics(CUBES[(model,block)],mask)
    for model in MODEL_FAMILIES for block in FEATURE_BLOCKS for group,mask in GROUP_MASKS.items()}


def information_flag(group, month):
    d = df.loc[GROUP_MASKS[group]]
    horizons = CFG.ibs_grid if month=="12_60" else [float(month)]
    if len(d)==0: return "EMPTY_GROUP"
    for h in horizons:
        cases = ((d.event==1)&(d.time_months<=h)).sum()
        controls = (d.time_months>h).sum()
        if cases<CFG.min_horizon_cases or controls<CFG.min_horizon_controls: return "LOW_INFORMATION"
    return "OK"


ones = np.ones(len(df),int)
OBSERVED = {key:metric_vector(value,ones) for key,value in PREPARED.items()}
performance_rows, diagnostic_rows = [], []
for (model,block,group), values in OBSERVED.items():
    for (month,metric), estimate in values.items():
        flag = information_flag(group,month) if np.isfinite(estimate) else "UNSUPPORTED_OR_UNDEFINED"
        performance_rows.append({"model":model,"block":block,"subgroup":group,"month":month,"metric":metric,
            "estimate":estimate,"n":int(GROUP_MASKS[group].sum()),"inference_flag":flag})
    d = PREPARED[(model,block,group)]
    for h in CFG.horizons:
        j = int(np.flatnonzero(PRED_TIMES==h)[0])
        for r in range(CFG.core_repeats):
            w = d["wc"][r,:,j]+d["wn"][r,:,j]
            diagnostic_rows.append({"model":model,"block":block,"subgroup":group,"repeat":r,"month":h,
                "cases":int(d["case"][r,:,j].sum()),"controls":int(d["control"][r,:,j].sum()),
                "min_G":float(d["G"][r,:,j].min()) if len(w) else np.nan,
                "max_weight":float(w.max()) if len(w) else np.nan,
                "effective_n":float(w.sum()**2/(w@w)) if w@w>0 else 0.,"supported":bool(d["valid"][r,j])})
performance_table = pd.DataFrame(performance_rows)
performance_table.to_csv(TABLE_DIR/"03_oof_performance.csv",index=False)
pd.DataFrame(diagnostic_rows).to_csv(TABLE_DIR/"03_evaluation_diagnostics.csv",index=False)

# Shared subject resamples preserve model pairing, repeat dependence and group relationships.
rng = np.random.default_rng(CFG.seed+717)
BOOT_DELTA = {(model,group):np.full((CFG.bootstrap_n,len(METRIC_KEYS)),np.nan)
    for model in MODEL_FAMILIES for group in GROUP_MASKS}
for b in tqdm(range(CFG.bootstrap_n),desc="Paired OOF bootstrap"):
    counts = np.bincount(rng.integers(0,len(df),len(df)),minlength=len(df))
    for model in MODEL_FAMILIES:
        for group in GROUP_MASKS:
            a = metric_vector(PREPARED[(model,"reference",group)],counts)
            z = metric_vector(PREPARED[(model,"extended",group)],counts)
            BOOT_DELTA[(model,group)][b] = [z[k]-a[k] for k in METRIC_KEYS]


def summarize_boot(observed, samples):
    x = np.asarray(samples,float)
    x = x[np.isfinite(x)]
    valid = len(x)>=int(np.ceil(CFG.min_boot_valid_fraction*CFG.bootstrap_n)) and np.isfinite(observed)
    if not valid:
        return {"estimate":observed,"ci_low":np.nan,"ci_high":np.nan,"p_boot_approx":np.nan,
                "n_boot_valid":len(x),"bootstrap_ok":False}
    p = min(1.,2*min((np.sum(x<=0)+1)/(len(x)+1),(np.sum(x>=0)+1)/(len(x)+1)))
    return {"estimate":observed,"ci_low":float(np.quantile(x,.025)),"ci_high":float(np.quantile(x,.975)),
            "p_boot_approx":float(p),"n_boot_valid":len(x),"bootstrap_ok":True}


def analysis_tier(group,month,metric):
    if group!="All" or month==12.: return "exploratory"
    if month==60. and metric=="AUC": return "primary"
    return "secondary"


delta_rows = []
for (model,group), raw in BOOT_DELTA.items():
    for j,(month,metric) in enumerate(METRIC_KEYS):
        a = OBSERVED[(model,"reference",group)][(month,metric)]
        b = OBSERVED[(model,"extended",group)][(month,metric)]
        row = summarize_boot(b-a,raw[:,j])
        flag = information_flag(group,month)
        if not np.isfinite(b-a): flag="UNSUPPORTED_OR_UNDEFINED"
        elif not row["bootstrap_ok"]: flag="INSUFFICIENT_VALID_BOOTSTRAPS"
        delta_rows.append({"model":model,"subgroup":group,"month":month,"metric":metric,
            "contrast":"extended_minus_reference","tier":analysis_tier(group,month,metric),"inference_flag":flag,**row})
delta_table = pd.DataFrame(delta_rows)


def adjust_pvalues(values, method):
    p = np.asarray(values,float)
    finite = np.isfinite(p)
    filled = np.where(finite,p,1.)
    m = len(p)
    if m==0: return p
    order = np.argsort(filled)
    x = filled[order]
    if method=="holm": out=np.maximum.accumulate(x*(m-np.arange(m)))
    elif method=="bh": out=np.minimum.accumulate((x*m/np.arange(1,m+1))[::-1])[::-1]
    else: raise ValueError(method)
    ans=np.empty(m);ans[order]=np.minimum(out,1.);ans[~finite]=np.nan
    return ans


delta_table["p_adjusted"] = np.nan
delta_table["adjustment_family"] = ""
for tier,method in [("primary","holm"),("secondary","bh")]:
    idx=delta_table.index[delta_table.tier.eq(tier)]
    delta_table.loc[idx,"p_adjusted"]=adjust_pvalues(delta_table.loc[idx,"p_boot_approx"],method)
    delta_table.loc[idx,"adjustment_family"]=tier+"_"+method
# Twelve-month estimates are information checks only; no additional inference family.
delta_table["statistically_supported"] = ((delta_table.ci_low>0)|(delta_table.ci_high<0)) & delta_table.p_adjusted.lt(.05) & delta_table.inference_flag.eq("OK")
delta_table["supported_improvement"] = delta_table.statistically_supported & np.where(
    delta_table.metric.eq("AUC"),delta_table.ci_low.gt(0),delta_table.ci_high.lt(0))
delta_table.to_csv(TABLE_DIR/"04_paired_deltas.csv",index=False)

np.savez_compressed(CHECKPOINT_DIR/f"bootstrap_deltas_B{CFG.bootstrap_n}.npz",
    **{f"{m}__{g}":a for (m,g),a in BOOT_DELTA.items()})
atomic_json(CHECKPOINT_DIR/f"bootstrap_deltas_B{CFG.bootstrap_n}.json",
    {"metric_order":METRIC_KEYS,"seed":CFG.seed+717,"training_signature":TRAINING_SIGNATURE,
     "inference":"paired subject bootstrap conditional on saved OOF predictions and censoring weights"})
display(delta_table.query("subgroup == 'All'"))

def calibration_fit(p,y,w):
    p=np.clip(np.asarray(p,float),1e-5,1-1e-5)
    y,w=np.asarray(y,float),np.asarray(w,float)
    if len(y)<3 or len(np.unique(y))<2 or np.std(p)<1e-10: raise ValueError("Insufficient calibration variation")
    lp=np.log(p/(1-p));X=np.column_stack([np.ones(len(p)),lp]);w=w/w.mean()
    def fit(offset,design,initial):
        def loss(beta):
            eta=offset+design@beta
            return float(np.sum(w*(np.logaddexp(0,eta)-y*eta)))
        def jac(beta): return design.T@(w*(expit(offset+design@beta)-y))
        opt=minimize(loss,initial,jac=jac,method="BFGS")
        if not opt.success or not np.isfinite(opt.x).all(): raise ValueError("Calibration fit did not converge: "+str(opt.message))
        return opt.x
    joint=fit(np.zeros(len(p)),X,np.array([0.,1.]))
    citl=fit(lp,np.ones((len(p),1)),np.array([0.]))[0]
    return {"joint_intercept":float(joint[0]),"slope":float(joint[1]),"calibration_in_the_large":float(citl)}


cal_rows=[]
for (model,block),cube in CUBES.items():
    for h in CFG.horizons:
        j=int(np.flatnonzero(PRED_TIMES==h)[0])
        for r in range(CFG.core_repeats):
            row={"model":model,"block":block,"month":h,"repeat":r}
            try:
                if not np.all((cube["supported"][r,:,j]==1)&np.isfinite(cube["surv"][r,:,j])):
                    raise ValueError("Unsupported prediction/censoring horizon")
                w=cube["wc"][r,:,j]+cube["wn"][r,:,j];use=w>0
                p=1-cube["surv"][r,use,j];y=cube["case"][r,use,j];ww=w[use]
                row.update(calibration_fit(p,y,ww));row["status"]="OK"
            except ValueError as exc: row.update({"status":"UNAVAILABLE","message":str(exc)})
            cal_rows.append(row)
calibration_table=pd.DataFrame(cal_rows)
calibration_table.to_csv(TABLE_DIR/"06_calibration_by_repeat.csv",index=False)
cal_summary=[]
for (model,block,h),g in calibration_table.groupby(["model","block","month"]):
    ok=g.status.eq("OK").all()
    row={"model":model,"block":block,"month":h,"valid_repeats":int(g.status.eq("OK").sum()),
        "status":"DESCRIPTIVE" if ok else "INCOMPLETE", "inference_flag":information_flag("All",h)}
    for c in ["joint_intercept","slope","calibration_in_the_large"]:
        row[c]=float(g[c].mean()) if ok and c in g else np.nan
    cal_summary.append(row)
pd.DataFrame(cal_summary).to_csv(TABLE_DIR/"06_calibration_summary.csv",index=False)

if (delta_table.n_boot_valid != CFG.bootstrap_n).any():
    raise RuntimeError("Incomplete prediction bootstrap")
if not calibration_table.loc[calibration_table.month.isin([36.,60.]),"status"].eq("OK").all():
    raise RuntimeError("Incomplete descriptive calibration")
