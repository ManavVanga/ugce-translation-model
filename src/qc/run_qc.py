import os
import json
import pandas as pd
from datetime import datetime

from src.utils.config import get_base_output_dir
from src.qc.qc_engine import run_qc


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

    print("Loading master row dataset...")

    if not os.path.exists(master_path) or os.path.getsize(master_path) == 0:
        print("Master dataset missing or empty. Writing empty QC outputs.")

        empty_df = pd.DataFrame(columns=QC_COLUMNS)

        empty_df.to_csv(qc_path, index=False)
        empty_df.to_csv(pass_path, index=False)
        empty_df.to_csv(review_path, index=False)
        empty_df.to_csv(fail_path, index=False)

        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "NO_MASTER_ROWS",
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

        print("QC completed with no master rows.")
        raise SystemExit(0)

    try:
        df = pd.read_csv(master_path)
    except pd.errors.EmptyDataError:
        print("Master file has no columns. Writing empty QC outputs.")

        empty_df = pd.DataFrame(columns=QC_COLUMNS)

        empty_df.to_csv(qc_path, index=False)
        empty_df.to_csv(pass_path, index=False)
        empty_df.to_csv(review_path, index=False)
        empty_df.to_csv(fail_path, index=False)

        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "EMPTY_MASTER_FILE",
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

        print("QC completed with empty master file.")
        raise SystemExit(0)

    print("Master rows:", len(df))

    if len(df) == 0:
        print("Master dataset has zero rows. Writing empty QC outputs.")

        empty_df = df.copy()
        for col in QC_COLUMNS:
            if col not in empty_df.columns:
                empty_df[col] = ""

        empty_df.to_csv(qc_path, index=False)
        empty_df.to_csv(pass_path, index=False)
        empty_df.to_csv(review_path, index=False)
        empty_df.to_csv(fail_path, index=False)

        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "ZERO_MASTER_ROWS",
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

        print("QC completed with zero rows.")
        raise SystemExit(0)

    print("Running QC...")
    qc_df = run_qc(df)

    qc_df.to_csv(qc_path, index=False)

    pass_df = qc_df[qc_df["qc_overall_status"] == "PASS"].copy()
    review_df = qc_df[qc_df["qc_overall_status"] == "REVIEW"].copy()
    fail_df = qc_df[qc_df["qc_overall_status"] == "FAIL"].copy()

    pass_df.to_csv(pass_path, index=False)
    review_df.to_csv(review_path, index=False)
    fail_df.to_csv(fail_path, index=False)

    total = len(qc_df)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "QC_COMPLETE",
        "total_rows": int(total),
        "pass_rows": int(len(pass_df)),
        "review_rows": int(len(review_df)),
        "fail_rows": int(len(fail_df)),
        "pass_rate_percent": round((len(pass_df) / total) * 100, 2) if total else 0,
        "review_rate_percent": round((len(review_df) / total) * 100, 2) if total else 0,
        "fail_rate_percent": round((len(fail_df) / total) * 100, 2) if total else 0,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    qc_df.to_csv(os.path.join(runs_dir, "translation_dataset_qc_v1.csv"), index=False)
    pass_df.to_csv(os.path.join(runs_dir, "translation_dataset_pass_v1.csv"), index=False)
    review_df.to_csv(os.path.join(runs_dir, "translation_dataset_review_v1.csv"), index=False)
    fail_df.to_csv(os.path.join(runs_dir, "translation_dataset_fail_v1.csv"), index=False)

    with open(os.path.join(runs_dir, "translation_dataset_qc_summary_v1.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved QC dataset:", qc_path)
    print("Saved PASS dataset:", pass_path)
    print("Saved REVIEW dataset:", review_path)
    print("Saved FAIL dataset:", fail_path)
    print("Saved QC summary:", summary_path)
    print(summary)
