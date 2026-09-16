"""Run the statistical analyses reported in the manuscript."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parent
SECTIONS = ("primary", "directional_outliers", "structural", "technical_risk",
            "reorientation_scalar", "competition", "spline", "prediction")
CORE = "baseline_clinical_alps.xlsx"
F2 = "reoriented_clinical_alps.xlsx"
INPUTS = {
    "primary": [CORE], "directional_outliers": [CORE],
    "structural": [CORE, "structural_mri_volumes.csv"],
    "technical_risk": [F2, "diffusion_motion_metrics.csv", "head_orientation_angles.csv",
                       "acquisition_plane_angles.csv"],
    "reorientation_scalar": [F2], "competition": [CORE], "spline": [F2], "prediction": [CORE],
}
DEFAULT_SETTINGS = {"risk_bootstrap": 500, "reorientation_bootstrap": 1000,
                    "competition_bootstrap": 300, "prediction_bootstrap": 300,
                    "device": "cpu", "prediction_cache": None}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_inputs(data_dir, sections):
    import numpy as np
    import pandas as pd
    names = sorted({name for section in sections for name in INPUTS[section]})
    missing = [name for name in names if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Required private input files are missing: " + ", ".join(missing)
                                + ". See docs/DATA_DICTIONARY.md.")
    common = ["AGE_INDEX", "SEX(M = 0)", "EDUCATION", "BASE_MMSE", "TIME_BY_DAYS",
              "EVENT_ADconversion", "WMH", "BRAIN_PARENCHYMA_FRACTION", "alps_L", "alps_R", "ALPS"]
    frames = []
    for name in (CORE, F2):
        if name not in names:
            continue
        d = pd.read_excel(data_dir / name, sheet_name="survival_dataset_final_age")
        if len(d) != 1131 or d.PTID.isna().any() or d.PTID.duplicated().any():
            raise ValueError(f"{name}: expected 1,131 unique participants")
        if not np.isfinite(d[common].to_numpy(float)).all():
            raise ValueError(f"{name}: incomplete or nonfinite primary inputs")
        if not d.EVENT_ADconversion.isin([0, 1]).all() or int(d.EVENT_ADconversion.sum()) != 115:
            raise ValueError(f"{name}: expected binary AD outcome and 115 conversions")
        if not d["SEX(M = 0)"].isin([0, 1]).all():
            raise ValueError("Sex must be male=0 / female=1")
        if not d.BASE_DX.isin(["CN", "MCI"]).all():
            raise ValueError("Baseline diagnosis must be CN or MCI")
        if not d.TIME_BY_DAYS.gt(0).all() or d.WMH.lt(0).any():
            raise ValueError("Follow-up must be positive; WMH must be nonnegative")
        if not d[["alps_L", "alps_R", "ALPS"]].gt(0).all().all():
            raise ValueError("ALPS values must be positive")
        np.testing.assert_allclose(d.ALPS, (d.alps_L + d.alps_R) / 2, atol=1e-12, rtol=1e-12)
        frames.append(d.set_index("PTID").sort_index())
    if len(frames) == 2:
        pd.testing.assert_frame_equal(frames[0][common], frames[1][common], check_dtype=False)
    return [{"filename": name, "sha256": sha256(data_dir / name)} for name in names]


def run(data_dir, output_dir, sections=SECTIONS, *, settings=None, check_only=False):
    """Public Python entry point, also used by the notebook."""
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    sections = tuple(sections)
    if not sections or any(s not in SECTIONS for s in sections):
        raise ValueError("Unknown or empty analysis section list")
    # Keep generated subject-level outputs outside the private input directory.
    if output_dir == data_dir or output_dir.is_relative_to(data_dir):
        raise ValueError("Output must be outside the input data directory")
    cfg = DEFAULT_SETTINGS | (settings or {})
    inputs = check_inputs(data_dir, sections)
    if check_only:
        print(f"Input checks passed for {len(inputs)} files.")
        return
    import matplotlib
    matplotlib.use("Agg")
    output_dir.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ("numpy", "pandas", "scipy", "lifelines", "statsmodels", "matplotlib",
                 "openpyxl", "scikit-learn", "torch", "torchtuples", "pycox"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "packages": packages, "settings": cfg, "inputs": inputs, "sections": [], "status": "running"}
    manifest_path = output_dir / "run_manifest.json"
    def checkpoint():
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    checkpoint()
    try:
        for section in sections:
            target = output_dir / section
            target.mkdir(exist_ok=True)
            source = ROOT / "analysis" / f"{section}.py"
            start = time.perf_counter()
            print(f"[{section}] running; log: {target / 'execution.log'}", flush=True)
            with (target / "execution.log").open("w", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    runpy.run_path(str(source), init_globals={"DATA_DIR": data_dir, "SECTION_OUT": target,
                                                             "SETTINGS": cfg})
            manifest["sections"].append({"name": section, "status": "complete", "source_sha256": sha256(source),
                                         "elapsed_seconds": round(time.perf_counter() - start, 2)})
            checkpoint()
            print(f"[{section}] complete ({manifest['sections'][-1]['elapsed_seconds']} s)", flush=True)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--sections", nargs="+", choices=SECTIONS, default=SECTIONS)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--prediction-cache", type=Path,
                        help="Optional saved OOF prediction directory; validated read-only evaluation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.sections, check_only=args.check_inputs,
        settings={"device": args.device, "prediction_cache": str(args.prediction_cache.resolve()) if args.prediction_cache else None})


if __name__ == "__main__":
    main()
