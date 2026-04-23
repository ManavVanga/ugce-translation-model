import os
import json
import pandas as pd
from datetime import datetime
from src.utils.config import get_base_output_dir
from src.extraction.deduplicator import deduplicate_against_existing, split_new_vs_duplicate


MASTER_FILENAME = "translation_row_master_v1.csv"
NEW_CANDIDATE_FILENAME = "translation_row_candidates_v1.csv"
DUPLICATE_FILENAME = "translation_row_duplicates_v1.csv"
APPEND_SUMMARY_FILENAME = "translation_row_append_summary_v1.json"


if __name__ == "__main__":
    base_dir = get_base_output_dir()

    collection_dir = os.path.join(base_dir, "collection")
    logs_dir = os.path.join(base_dir, "logs")
    runs_dir = os.path.join(base_dir, "runs", f"append_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    os.makedirs(collection_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    candidate_path = os.path.join(collection_dir, NEW_CANDIDATE_FILENAME)
    master_path = os.path.join(collection_dir, MASTER_FILENAME)
    duplicate_path = os.path.join(collection_dir, DUPLICATE_FILENAME)
    summary_path = os.path.join(collection_dir, APPEND_SUMMARY_FILENAME)
    log_path = os.path.join(logs_dir, "data_collection_log.csv")

    run_master_snapshot = os.path.join(runs_dir, MASTER_FILENAME)
    run_new_snapshot = os.path.join(runs_dir, NEW_CANDIDATE_FILENAME)
    run_duplicate_snapshot = os.path.join(runs_dir, DUPLICATE_FILENAME)
    run_summary_snapshot = os.path.join(runs_dir, APPEND_SUMMARY_FILENAME)

    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Candidate row file not found: {candidate_path}")

    print("Loading candidate rows...")
    candidate_df = pd.read_csv(candidate_path)

    if os.path.exists(master_path):
        print("Loading existing master dataset...")
        master_df = pd.read_csv(master_path)
    else:
        print("No existing master dataset found. Initializing new master.")
        master_df = pd.DataFrame(columns=candidate_df.columns)

    print("Deduplicating candidates against existing master...")
    checked_df = deduplicate_against_existing(candidate_df, master_df)
    new_only_df, duplicate_df = split_new_vs_duplicate(checked_df)

    # keep same columns in master; include dedup columns only in duplicate audit if you want
    master_append_df = new_only_df.drop(columns=["is_duplicate"], errors="ignore")
    master_append_df = master_append_df.drop(columns=["dedup_key"], errors="ignore")

    duplicate_save_df = duplicate_df.copy()

    updated_master_df = pd.concat([master_df, master_append_df], ignore_index=True)

    # final internal exact dedup on record_id if present
    if "record_id" in updated_master_df.columns:
        updated_master_df = updated_master_df.drop_duplicates(subset=["record_id"], keep="first").reset_index(drop=True)

    updated_master_df.to_csv(master_path, index=False)
    duplicate_save_df.to_csv(duplicate_path, index=False)
    candidate_df.to_csv(run_new_snapshot, index=False)
    updated_master_df.to_csv(run_master_snapshot, index=False)
    duplicate_save_df.to_csv(run_duplicate_snapshot, index=False)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_rows_in_run": int(len(candidate_df)),
        "new_rows_appended": int(len(master_append_df)),
        "duplicate_rows_skipped": int(len(duplicate_save_df)),
        "master_total_rows_after_append": int(len(updated_master_df))
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(run_summary_snapshot, "w") as f:
        json.dump(summary, f, indent=2)

    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
    else:
        log_df = pd.DataFrame(columns=[
            "log_id", "date_collected", "dataset_name", "source_type",
            "study_type", "num_records_added", "curated_by", "notes"
        ])

    new_log = pd.DataFrame([{
        "log_id": f"LOG_{len(log_df)+1:03d}",
        "date_collected": datetime.now().strftime("%Y-%m-%d"),
        "dataset_name": MASTER_FILENAME,
        "source_type": "dedup_append_pipeline",
        "study_type": "row_level_translation_records",
        "num_records_added": int(len(master_append_df)),
        "curated_by": "auto_append_v1",
        "notes": f"duplicates_skipped={len(duplicate_save_df)}"
    }])

    log_df = pd.concat([log_df, new_log], ignore_index=True)
    log_df.to_csv(log_path, index=False)

    print("Append complete")
    print("Master file:", master_path)
    print("Duplicate audit file:", duplicate_path)
    print("Summary:", summary)
