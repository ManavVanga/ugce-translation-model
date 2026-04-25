# src/qc/gdsc_qc.py

import os
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd


QC_VERSION = "GDSC_QC_v1"


def is_missing(x):
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def safe_float(x):
    try:
        if is_missing(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_flag(df, mask, flag, severity):
    mask = mask.fillna(False)

    df.loc[mask, "gdsc_qc_flags"] = (
        df.loc[mask, "gdsc_qc_flags"].astype(str) + flag + ";"
    )

    if severity == "FAIL":
        df.loc[mask, "gdsc_qc_status"] = "FAIL"
        df.loc[mask, "gdsc_qc_fail_reason"] = (
            df.loc[mask, "gdsc_qc_fail_reason"].astype(str) + flag + ";"
        )

    elif severity == "REVIEW":
        review_mask = mask & (df["gdsc_qc_status"] != "FAIL")
        df.loc[review_mask, "gdsc_qc_status"] = "REVIEW"
        df.loc[review_mask, "gdsc_qc_review_reason"] = (
            df.loc[review_mask, "gdsc_qc_review_reason"].astype(str) + flag + ";"
        )

    return df


def run_gdsc_qc(df):
    df = df.copy()

    df["gdsc_qc_version"] = QC_VERSION
    df["gdsc_qc_status"] = "PASS"
    df["gdsc_qc_flags"] = ""
    df["gdsc_qc_fail_reason"] = ""
    df["gdsc_qc_review_reason"] = ""

    required_cols = [
        "record_id",
        "drug_name_standard",
        "vitro_system_class",
        "assay_type",
        "assay_endpoint",
        "response_value_standard",
        "response_metric_standard",
        "outcome_level",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
            df = add_flag(df, pd.Series(True, index=df.index), f"MISSING_COLUMN_{col}", "FAIL")

    # Required source-level fields
    for col in required_cols:
        df = add_flag(
            df,
            df[col].apply(is_missing),
            f"MISSING_REQUIRED_{col}",
            "FAIL",
        )

    # GDSC should be in vitro
    df = add_flag(
        df,
        df["outcome_level"].astype(str).str.strip().str.lower() != "in_vitro",
        "INVALID_GDSC_OUTCOME_LEVEL_NOT_IN_VITRO",
        "FAIL",
    )

    # Drug sensitivity response must be numeric
    df["_gdsc_response_numeric"] = df["response_value_standard"].apply(safe_float)

    df = add_flag(
        df,
        df["_gdsc_response_numeric"].isna(),
        "RESPONSE_NOT_NUMERIC",
        "FAIL",
    )

    # Biological range checks
    metric = df["response_metric_standard"].astype(str).str.strip()

    ic50_mask = metric.isin(["IC50_uM", "IC50", "ic50_uM", "ic50"])
    ln_ic50_mask = metric.isin(["LN_IC50", "ln_IC50", "ln_ic50"])
    viability_mask = metric.isin(["viability_percent", "VIABILITY_PERCENT"])
    auc_mask = metric.isin(["AUC", "auc"])

    df = add_flag(
        df,
        ic50_mask & ~df["_gdsc_response_numeric"].between(0, 10000, inclusive="neither"),
        "IC50_OUT_OF_RANGE",
        "FAIL",
    )

    df = add_flag(
        df,
        ln_ic50_mask & ~df["_gdsc_response_numeric"].between(-20, 20, inclusive="both"),
        "LN_IC50_OUT_OF_RANGE",
        "FAIL",
    )

    df = add_flag(
        df,
        viability_mask & ~df["_gdsc_response_numeric"].between(0, 100, inclusive="both"),
        "VIABILITY_PERCENT_OUT_OF_RANGE",
        "FAIL",
    )

    df = add_flag(
        df,
        auc_mask & ~df["_gdsc_response_numeric"].between(0, 1.5, inclusive="both"),
        "AUC_OUT_OF_RANGE",
        "FAIL",
    )

    # Unknown response metric = review, not fail
    known_metrics = ic50_mask | ln_ic50_mask | viability_mask | auc_mask
    df = add_flag(
        df,
        ~known_metrics,
        "UNKNOWN_RESPONSE_METRIC_REVIEW",
        "REVIEW",
    )

    # Optional but important fields
    optional_review_cols = [
        "tissue_context",
        "dose_normalized_uM",
        "exposure_time_hours",
        "cell_line_name",
        "model_name",
        "source_name",
    ]

    for col in optional_review_cols:
        if col in df.columns:
            df = add_flag(
                df,
                df[col].apply(is_missing),
                f"MISSING_OPTIONAL_{col}",
                "REVIEW",
            )

    # Duplicate checks
    dup_cols = [
        c for c in [
            "drug_name_standard",
            "cell_line_name",
            "model_name",
            "response_metric_standard",
            "response_value_standard",
        ]
        if c in df.columns
    ]

    if len(dup_cols) >= 3:
        df = add_flag(
            df,
            df.duplicated(subset=dup_cols, keep="first"),
            "EXACT_DUPLICATE_REVIEW",
            "REVIEW",
        )

    conflict_cols = [
        c for c in ["drug_name_standard", "cell_line_name", "model_name", "response_metric_standard"]
        if c in df.columns
    ]

    if len(conflict_cols) >= 3:
        conflict = (
            df.groupby(conflict_cols)["_gdsc_response_numeric"]
            .nunique(dropna=True)
            .reset_index(name="n_unique_response")
        )

        conflict_keys = conflict[conflict["n_unique_response"] > 1][conflict_cols]

        if len(conflict_keys) > 0:
            conflict_keys["_conflict"] = 1
            temp = df.merge(conflict_keys, on=conflict_cols, how="left")
            df = add_flag(
                df,
                temp["_conflict"].eq(1),
                "CONFLICTING_DUPLICATE_RESPONSE_REVIEW",
                "REVIEW",
            )

    return df


def write_gdsc_qc_outputs(qc_df, out_dir, input_path=None):
    os.makedirs(out_dir, exist_ok=True)

    pass_df = qc_df[qc_df["gdsc_qc_status"] == "PASS"].copy()
    review_df = qc_df[qc_df["gdsc_qc_status"] == "REVIEW"].copy()
    fail_df = qc_df[qc_df["gdsc_qc_status"] == "FAIL"].copy()

    qc_path = os.path.join(out_dir, "gdsc_qc_full_v1.csv")
    pass_path = os.path.join(out_dir, "gdsc_pass_training_grade_v1.csv")
    review_path = os.path.join(out_dir, "gdsc_review_v1.csv")
    fail_path = os.path.join(out_dir, "gdsc_fail_v1.csv")
    summary_path = os.path.join(out_dir, "gdsc_qc_summary_v1.json")

    qc_df.to_csv(qc_path, index=False)
    pass_df.to_csv(pass_path, index=False)
    review_df.to_csv(review_path, index=False)
    fail_df.to_csv(fail_path, index=False)

    flag_counts = {}
    for flags in qc_df["gdsc_qc_flags"].fillna("").astype(str):
        for f in flags.split(";"):
            f = f.strip()
            if f:
                flag_counts[f] = flag_counts.get(f, 0) + 1

    total = len(qc_df)

    summary = {
        "qc_version": QC_VERSION,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": input_path,
        "input_sha256": sha256_file(input_path) if input_path and os.path.exists(input_path) else None,
        "total_rows": total,
        "pass_rows": len(pass_df),
        "review_rows": len(review_df),
        "fail_rows": len(fail_df),
        "pass_rate_percent": round((len(pass_df) / total) * 100, 2) if total else 0,
        "review_rate_percent": round((len(review_df) / total) * 100, 2) if total else 0,
        "fail_rate_percent": round((len(fail_df) / total) * 100, 2) if total else 0,
        "flag_counts": flag_counts,
        "outputs": {
            "qc_full": qc_path,
            "pass": pass_path,
            "review": review_path,
            "fail": fail_path,
            "summary": summary_path,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    return summary
