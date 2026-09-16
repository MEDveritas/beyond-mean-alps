"""Compare recomputed aggregate statistics with study reference results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def compare(results, *, include_prediction=True):
    manifest=json.loads((ROOT/"reference"/"manifest.json").read_text(encoding="utf-8"))
    checks=[]
    for spec in manifest:
        if not include_prediction and spec["result"].startswith("prediction/"):
            continue
        expected=pd.read_csv(ROOT/"reference"/spec["reference"])
        actual=pd.read_csv(Path(results)/spec["result"])
        keys=spec["keys"]
        for name in keys:
            expected[name]=expected[name].astype(str).str.replace(r"\.0$","",regex=True)
            actual[name]=actual[name].astype(str).str.replace(r"\.0$","",regex=True)
        if keys:
            if expected.duplicated(keys).any() or actual.duplicated(keys).any():
                raise ValueError("Nonunique result keys: "+spec["result"])
            aligned=expected[keys].merge(actual,on=keys,how="left",validate="one_to_one",indicator=True)
            if not aligned._merge.eq("both").all():
                raise AssertionError("Missing reference rows: "+spec["result"])
        else:
            if len(actual)!=len(expected):
                raise AssertionError("Row-count mismatch: "+spec["result"])
            aligned=actual
        for column in spec["numeric_columns"]:
            a=pd.to_numeric(aligned[column]).to_numpy(float)
            b=pd.to_numeric(expected[column]).to_numpy(float)
            passed=bool(np.allclose(a,b,atol=1e-8,rtol=1e-7,equal_nan=True))
            finite=np.isfinite(a)&np.isfinite(b)
            checks.append(dict(result=spec["result"],column=column,rows=len(b),passed=passed,
                               maximum_absolute_error=float(np.abs(a[finite]-b[finite]).max()) if finite.any() else 0.))
    report=pd.DataFrame(checks)
    report.to_csv(Path(results)/"reference_comparison.csv",index=False)
    failed=report.loc[~report.passed]
    print(f"Reference comparison: {int(report.passed.sum())}/{len(report)} column checks passed")
    if len(failed):
        print(failed.to_string(index=False))
        raise AssertionError("Results differ from the study reference statistics")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=Path("results"))
    p.add_argument("--skip-prediction",action="store_true",
                   help="Compare association/sensitivity results only after a fresh neural retraining run")
    a=p.parse_args()
    compare(a.results,include_prediction=not a.skip_prediction)
