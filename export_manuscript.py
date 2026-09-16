"""Export manuscript tables and figure source values from analysis results.

Tables are assembled from completed model outputs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd


def export(data_dir, results):
    data_dir, results = Path(data_dir), Path(results)
    out = results / "manuscript"
    out.mkdir(parents=True, exist_ok=True)
    def read(section, filename):
        return pd.read_csv(results / section / filename)
    def save(frame, name):
        frame.to_csv(out / name, index=False)
    d = pd.read_excel(data_dir / "baseline_clinical_alps.xlsx", sheet_name="survival_dataset_final_age")
    d["absolute_asymmetry"] = (d.alps_L-d.alps_R).abs()
    d["BPF_percent"] = 100*d.BRAIN_PARENCHYMA_FRACTION
    rows = []
    for group, frame in [("All",d),("CN",d.loc[d.BASE_DX.eq("CN")]),("MCI",d.loc[d.BASE_DX.eq("MCI")])]:
        for name in ["AGE_INDEX","EDUCATION","BASE_MMSE","BPF_percent","ALPS","absolute_asymmetry","WMH","TIME_BY_MONTHS"]:
            x=pd.to_numeric(frame[name],errors="raise")
            rows.append(dict(group=group,variable=name,n=len(x),mean=x.mean(),SD=x.std(ddof=1),
                             median=x.median(),Q1=x.quantile(.25),Q3=x.quantile(.75)))
        for name in ["SEX(M = 0)","EVENT_ADconversion"]:
            rows.append(dict(group=group,variable=name,n=len(frame),count=int(frame[name].sum()),percent=100*frame[name].mean()))
    save(pd.DataFrame(rows),"Table1_characteristics.csv")
    save(read("primary","04_primary_Cox_full_cohort.csv"),"Table2_primary_Cox.csv")
    # Table S1 counts are a transcription of the manuscript, not reconstructed exclusions.
    flow=pd.DataFrame([
        (0,"Eligible diffusion MRI and dated diagnostic record",0,1846),
        (1,"No diagnosis within 90 days",323,1523),
        (2,"AD at baseline",178,1345),
        (3,"No post-baseline diagnostic assessment",141,1204),
        (4,"Baseline MMSE unavailable",2,1202),
        (5,"Fewer than 30 diffusion directions",8,1194),
        (6,"Required structural volumes unavailable",47,1147),
        (7,"ALPS processing failure",16,1131),
    ],columns=["step","criterion","excluded","remaining"])
    flow["source"]="Manuscript Table S1; reported cohort-selection counts"
    save(flow,"TableS1_reported_selection_counts.csv")
    save(pd.concat([read("directional_outliers","03_left_right_signed_absolute_Cox.csv"),
                    read("directional_outliers","07_tensor_component_negative_controls.csv")],ignore_index=True),"TableS2_directional_components.csv")
    save(pd.concat([read("directional_outliers","10_outlier_tail_exclusion_Cox.csv"),
                    read("directional_outliers","12_influence_sensitivity_Cox.csv")],ignore_index=True),"TableS3_outliers_influence.csv")
    for src,dst in [
        ("03_absALPS_structural_correlations.csv","TableS4_correlations.csv"),
        ("05_structural_asymmetry_Cox_ladder.csv","TableS4_same_day_Cox.csv"),
        ("10_365day_structural_sensitivity.csv","TableS4_365day_Cox.csv")]:
        save(read("structural",src),dst)
    for section,src,dst in [
        ("reorientation_scalar","02_f1_f2_reproducibility.csv","TableS5A_agreement.csv"),
        ("reorientation_scalar","03_f1_f2_same_specification_cox.csv","TableS5B_Cox.csv"),
        ("reorientation_scalar","04_f1_f2_paired_bootstrap_summary.csv","TableS5B_bootstrap.csv"),
        ("reorientation_scalar","06_FA_MD_negative_control_cox.csv","TableS6_scalar_controls.csv"),
        ("technical_risk","04_technical_cox_model_comparison.csv","TableS7_technical.csv"),
        ("technical_risk","06_alternative_asymmetry_definitions.csv","TableS8_definitions.csv"),
        ("competition","03_primary_competition_coefficients.csv","TableS9A_competition.csv"),
        ("competition","04_primary_nested_LR_tests.csv","TableS9B_LR.csv"),
        ("competition","17_bootstrap_joint_model_summary.csv","TableS9B_bootstrap.csv"),
        ("spline","figure2b_spline_curve.csv","Figure3A_spline_curve.csv"),
        ("spline","figure2b_spline_metadata.csv","Figure3A_spline_tests.csv"),
        ("technical_risk","10_adjusted_36_60_month_absolute_risk.csv","Figure3BC_absolute_risk.csv"),
        ("directional_outliers","Figure4C_component_correlations.csv","Figure4C_component_correlations.csv"),
        ("competition","05_coefficient_attenuation.csv","Figure4C_attenuation.csv")]:
        save(read(section,src),dst)
    perf=read("prediction/tables","03_oof_performance.csv")
    delta=read("prediction/tables","04_paired_deltas.csv")
    for frame in [perf,delta]:
        frame["month"]=frame.month.astype(str).str.replace(r"\.0$","",regex=True)
    values=perf.loc[perf.subgroup.eq("All")].pivot(index=["model","month","metric"],columns="block",values="estimate").reset_index()
    joined=delta.loc[delta.subgroup.eq("All")].merge(values,on=["model","month","metric"],validate="one_to_one")
    auc=joined.loc[joined.metric.eq("AUC") & joined.month.isin(["36","60"])].copy()
    for c in ["estimate","ci_low","ci_high"]:auc[c+"_percentage_points"]=100*auc[c]
    save(auc,"TableS10_discrimination.csv")
    error=joined.loc[joined.metric.isin(["Brier","IBS"]) & joined.month.isin(["36","60","12_60"])].copy()
    for c in ["reference","extended","estimate","ci_low","ci_high"]:error[c+"_times_1000"]=1000*error[c]
    save(error,"TableS11_prediction_error.csv")
    cal=read("prediction/tables","06_calibration_summary.csv")
    save(cal.loc[cal.month.isin([36,60])],"TableS12_calibration.csv")
    (out/"README.txt").write_text(
        "Table1 follow-up uses TIME_BY_MONTHS. Survival models use TIME_BY_DAYS.\n"
        "TableS1 contains the cohort-selection counts reported in the manuscript.\n"
        "Other files contain calculated statistics from the analysis modules.\n"
        "Exported filenames follow the manuscript table and figure numbering.\n",encoding="utf-8")
    print(f"Exported manuscript tables and figure source data: {out}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=Path("data"))
    p.add_argument("--results",type=Path,default=Path("results"))
    a=p.parse_args()
    export(a.data_dir,a.results)
