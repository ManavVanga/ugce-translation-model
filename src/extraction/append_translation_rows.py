import os
import json
import pandas as pd
from datetime import datetime
from src.utils.config import get_base_output_dir


if __name__ == "__main__":
    base_dir = get_base_output_dir()

    collection_dir = os.path.join(base_dir, "collection")
    runs_dir = os.path.join(base_dir, "runs", f"append_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    os.makedirs(collection_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    candidate_path = os.path.join(collection_dir, "translation_row_candidates_v1.csv")
    master_path = os.path.join(collection_dir, "translation_row_master_v1.csv")
    duplicates_path = os.path.join(collection_dir, "translation_row_duplicates_v1.csv")
    summary_path = os.path.join(collection_dir, "translation_row_append_summary_v1.json")

    print("Loading candidate rows...")

    if not os.path.exists(candidate_path) or os.path.getsize(candidate_path) == 0:
        print("No candidate rows found. Writing empty master/duplicate files.")

        empty_df = pd.DataFrame()
        empty_df.to_csv(master_path, index=False)
        empty_df.to_csv(duplicates_path, index=False)

        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "NO_CANDIDATE_ROWS",
            "candidate_rows": 0,
            "existing_master_rows": 0,
            "new_master_rows": 0,
            "duplicate_rows": 0
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        empty_df.to_csv(os.path.join(runs_dir, "translation_row_master_v1.csv"), index=False)
        empty_df.to_csv(os.path.join(runs_dir, "translation_row_duplicates_v1.csv"), index=False)

        with open(os.path.join(runs_dir, "translation_row_append_summary_v1.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("Append completed with no candidate rows.")
        raise SystemExit(0)

    try:
        candidate_df = pd.read_csv(candidate_path)
    except pd.errors.EmptyDataError:
        print("Candidate file exists but has no columns. Writing empty outputs.")

        empty_df = pd.DataFrame()
        empty_df.to_csv(master_path, index=False)
        empty_df.to_csv(duplicates_path, index=False)

        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "EMPTY_CANDIDATE_FILE",
            "candidate_rows": 0,
            "existing_master_rows": 0,
            "new_master_rows": 0,
            "duplicate_rows": 0
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("Append completed with empty candidate file.")
        raise SystemExit(0)

    print("Candidate rows:", len(candidate_df))

    if os.path.exists(master_path) and os.path.getsize(master_path) > 0:
        try:
            old_master_df = pd.read_csv(master_path)
        except pd.errors.EmptyDataError:
            old_master_df = pd.DataFrame()
    else:
        old_master_df = pd.DataFrame()

    print("Existing master rows:", len(old_master_df))

    if len(old_master_df) > 0:
        combined_df = pd.concat([old_master_df, candidate_df], ignore_index=True)
    else:
        combined_df = candidate_df.copy()

    dedup_cols = [c for c in [
        "paper_id",
        "pmid",
        "drug_name_standard",
        "vitro_system_class",
        "assay_type",
        "dose_normalized_uM",
        "exposure_time_hours",
        "response_value_standard",
        "response_metric_standard",
        "evidence_sentence"
    ] if c in combined_df.columns]

    if dedup_cols:
        duplicate_mask = combined_df.duplicated(subset=dedup_cols, keep="first")
        duplicates_df = combined_df[duplicate_mask].copy()
        master_df = combined_df[~duplicate_mask].copy()
    else:
        duplicates_df = pd.DataFrame()
        master_df = combined_df.copy()

    master_df.to_csv(master_path, index=False)
    duplicates_df.to_csv(duplicates_path, index=False)

    master_df.to_csv(os.path.join(runs_dir, "translation_row_master_v1.csv"), index=False)
    duplicates_df.to_csv(os.path.join(runs_dir, "translation_row_duplicates_v1.csv"), index=False)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "APPEND_COMPLETE",
        "candidate_rows": int(len(candidate_df)),
        "existing_master_rows": int(len(old_master_df)),
        "new_master_rows": int(len(master_df)),
        "duplicate_rows": int(len(duplicates_df)),
        "dedup_columns": dedup_cols
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(runs_dir, "translation_row_append_summary_v1.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved master:", master_path)
    print("Saved duplicates:", duplicates_path)
    print("Saved summary:", summary_path)
    print(summary)
