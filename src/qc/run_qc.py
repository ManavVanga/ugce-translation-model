import os
import json
import pandas as pd
from datetime import datetime
from src.utils.config import get_base_output_dir


QC_COLUMNS = [
    "qc_schema_pass",
    "qc_schema_reason",
    "qc_format_pass",
    "qc_format_reason",
    "qc_biological_pass",
    "qc_biological_reason",
    "qc_linkage_pass",
    "qc_linkage_reason",
    "qc_evidence_pass",
    "qc_evidence_reason",
    "qc_overall_status",
    "qc_overall_reason",
]


def is_filled(row, col):
    if col not in row.index:
        return False
    val = row[col]
    if pd.isna(val):
        return False
    return str(val).strip() != ""


def run_qc(df):
    qc_rows = []

    for _, row in df.iterrows():
        r = row.copy()

        required = [
            "record_id",
            "paper_id",
            "pmid",
            "drug_name_standard",
            "vitro_system_class",
            "assay_type",
            "dose_normalized_uM",
            "outcome_level",
        ]

        missing = [c for c in required if not is_filled(row, c)]

        response_present = (
            is_filled(row, "response_value_standard")
            and is_filled(row, "response_metric_standard")
        )

        evidence_present = (
            is_filled(row, "evidence_sentence")
            or is_filled(row, "evidence_window")
        )

        qc_schema_pass = len(missing) == 0
        qc_format_pass = True
        qc_biological_pass = response_present
        qc_linkage_pass = is_filled(row, "paper_id") and is_filled(row, "pmid")
        qc_evidence_pass = evidence_present

        r["qc_schema_pass"] = "PASS" if qc_schema_pass else "FAIL"
        r["qc_schema_reason"] = "" if qc_schema_pass else "missing:" + ",".join(missing)

        r["qc_format_pass"] = "PASS"
        r["qc_format_reason"] = ""

        r["qc_biological_pass"] = "PASS" if qc_biological_pass else "FAIL"
        r["qc_biological_reason"] = "" if qc_biological_pass else "missing_response_value_or_metric"

        r["qc_linkage_pass"] = "PASS" if qc_linkage_pass else "FAIL"
        r["qc_linkage_reason"] = "" if qc_linkage_pass else "missing_paper_or_pmid"

        r["qc_evidence_pass"] = "PASS" if qc_evidence_pass else "FAIL"
        r["qc_evidence_reason"] = "" if qc_evidence_pass else "missing_evidence_trace"

        if qc_schema_pass and qc_biological_pass and qc_linkage_pass and qc_evidence_pass:
            r["qc_overall_status"] = "PASS"
            r["qc_overall_reason"] = "training_ready"
        elif qc_linkage_pass and qc_evidence_pass:
            r["qc_overall_status"] = "REVIEW"
            reasons = []
            if not qc_schema_pass:
                reasons.append("schema_incomplete")
            if not qc_biological_pass:
                reasons.append("missing_response")
            r["qc_overall_reason"] = "|".join(reasons)
        else:
            r["qc_overall_status"] = "FAIL"
            r["qc_overall_reason"] = "insufficient_linkage_or_evidence"

        qc_rows.append(r)

    return pd.DataFrame(qc_rows)


def write_empty_outputs(curated_dir, runs_dir, summary_path, status):
    empty_df = pd.DataFrame(columns=QC_COLUMNS)

    qc_path = os.path.join(curated_dir, "translation_dataset_qc_v1.csv")
    pass_path = os.path.join(curated_dir, "translation_dataset_pass_v1.csv")
    review_path = os.path.join(curated_dir, "translation_dataset_review_v1.csv")
    fail_path = os.path.join(curated_dir, "translation_dataset_fail_v1.csv")

    empty_df.to_csv(qc_path, index=False)
    empty_df.to_csv(pass_path, index=False)
    empty_df.to_csv(review_path, index=False)
    empty_df.to_csv(fail_path, index=False)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "total_rows": 0,
        "pass_rows": 0,
        "review_rows": 0,
        "fail_rows": 0,
        "pass_rate_percent": 0,
        "review_rate_percent": 0,
        "fail_rate_percent": 0,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    empty_df.to_csv(os.path.join(runs_dir, "translation_dataset_qc_v1.csv"), index=False)
    empty_df.to_csv(os.path.join(runs_dir, "translation_dataset_pass_v1.csv"), index=False)
    empty_df.to_csv(os.path.join(runs_dir, "translation_dataset_review_v1.csv"), index=False)
    empty_df.to_csv(os.path.join(runs_dir, "translation_dataset_fail_v1.csv"), index=False)

    with open(os.path.join(runs_dir, "translation_dataset_qc_summary_v1.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(summary)


if __name__ == "__main__":
    base_dir = get_base_output_dir()

    collection_dir = os.path.join(base_dir, "collection")
    curated_dir = os.path.join(base_dir, "curated")
    runs_dir = os.path.join(base_dir, "runs", f"qc_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    os.makedirs(curated_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    master_path = os.path.join(collection_dir, "translation_row_master_v1.csv")

    qc_path = os.path.join(curated_dir, "translation_dataset_qc_v1.csv")
    pass_path = os.path.join(curated_dir, "translation_dataset_pass_v1.csv")
    review_path = os.path.join(curated_dir, "translation_dataset_review_v1.csv")
    fail_path = os.path.join(curated_dir, "translation_dataset_fail_v1.csv")
    summary_path = os.path.join(curated_dir, "translation_dataset_qc_summary_v1.json")

    print("Loading master dataset...")

    if not os.path.exists(master_path) or os.path.getsize(master_path) == 0:
        write_empty_outputs(curated_dir, runs_dir, summary_path, "NO_MASTER_ROWS")
        raise SystemExit(0)

    try:
        df = pd.read_csv(master_path)
    except pd.errors.EmptyDataError:
        write_empty_outputs(curated_dir, runs_dir, summary_path, "EMPTY_MASTER_FILE")
        raise SystemExit(0)

    if len(df) == 0:
        write_empty_outputs(curated_dir, runs_dir, summary_path, "ZERO_MASTER_ROWS")
        raise SystemExit(0)

    print("Running QC...")
    qc_df = run_qc(df)

    pass_df = qc_df[qc_df["qc_overall_status"] == "PASS"]
    review_df = qc_df[qc_df["qc_overall_status"] == "REVIEW"]
    fail_df = qc_df[qc_df["qc_overall_status"] == "FAIL"]

    qc_df.to_csv(qc_path, index=False)
    pass_df.to_csv(pass_path, index=False)
    review_df.to_csv(review_path, index=False)
    fail_df.to_csv(fail_path, index=False)

    total = len(qc_df)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "QC_COMPLETE",
        "total_rows": total,
        "pass_rows": len(pass_df),
        "review_rows": len(review_df),
        "fail_rows": len(fail_df),
        "pass_rate_percent": round((len(pass_df) / total) * 100, 2) if total else 0,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(summary)
