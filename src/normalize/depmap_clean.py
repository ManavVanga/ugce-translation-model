import os
import json
import hashlib
from datetime import datetime

import pandas as pd


RAW_DIR = "data_sources/depmap_ccle/raw"
OUT_DIR = "outputs/processed/depmap_ccle"

FILES = {
    "model": "Model.csv",
    "mutations": "OmicsSomaticMutations.csv",
    "expression": "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
    "crispr": "CRISPRGeneEffect.csv",
    "cnv": "PortalOmicsCNGeneLog2.csv",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clean_wide_matrix(input_path, output_path, duplicate_log_path, file_type):
    df = pd.read_csv(input_path, low_memory=False)

    original_rows = len(df)
    original_cols = df.shape[1]

    # Fix unnamed identifier column
    if "Unnamed: 0" in df.columns and "ModelID" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "ModelID"})

    if "ModelID" not in df.columns:
        raise ValueError(f"{file_type}: ModelID column not found after cleaning.")

    duplicate_mask = df.duplicated(subset=["ModelID"], keep=False)
    duplicate_df = df[duplicate_mask].copy()

    if len(duplicate_df) > 0:
        duplicate_df.to_csv(duplicate_log_path, index=False)

    # Strict and auditable policy:
    # keep first ModelID row, log all duplicated rows for review
    clean_df = df.drop_duplicates(subset=["ModelID"], keep="first").copy()

    clean_df.to_csv(output_path, index=False)

    return {
        "file_type": file_type,
        "input_path": input_path,
        "output_path": output_path,
        "duplicate_log_path": duplicate_log_path if len(duplicate_df) > 0 else None,
        "original_rows": int(original_rows),
        "original_columns": int(original_cols),
        "clean_rows": int(len(clean_df)),
        "clean_columns": int(clean_df.shape[1]),
        "duplicate_rows_logged": int(len(duplicate_df)),
        "unique_model_ids_after_cleaning": int(clean_df["ModelID"].nunique(dropna=True)),
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "cleaning_actions": [
            "renamed_Unnamed_0_to_ModelID_if_present",
            "logged_duplicate_ModelID_rows",
            "kept_first_ModelID_row_for_clean_dataset",
        ],
    }


def copy_long_table(input_path, output_path, file_type):
    df = pd.read_csv(input_path, low_memory=False)
    df.to_csv(output_path, index=False)

    return {
        "file_type": file_type,
        "input_path": input_path,
        "output_path": output_path,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "cleaning_actions": ["copied_without_row_filtering"],
    }


def run_depmap_clean(raw_dir=RAW_DIR, out_dir=OUT_DIR):
    ensure_dir(out_dir)

    duplicate_dir = os.path.join(out_dir, "duplicate_logs")
    ensure_dir(duplicate_dir)

    reports = []

    # Model metadata
    reports.append(
        copy_long_table(
            os.path.join(raw_dir, FILES["model"]),
            os.path.join(out_dir, "Model_clean.csv"),
            "model",
        )
    )

    # Mutation long table
    reports.append(
        copy_long_table(
            os.path.join(raw_dir, FILES["mutations"]),
            os.path.join(out_dir, "OmicsSomaticMutations_clean.csv"),
            "mutations",
        )
    )

    # Expression wide matrix
    reports.append(
        clean_wide_matrix(
            os.path.join(raw_dir, FILES["expression"]),
            os.path.join(out_dir, "OmicsExpressionTPMLogp1HumanProteinCodingGenes_clean.csv"),
            os.path.join(duplicate_dir, "expression_duplicate_ModelID_rows.csv"),
            "expression",
        )
    )

    # CRISPR wide matrix
    reports.append(
        clean_wide_matrix(
            os.path.join(raw_dir, FILES["crispr"]),
            os.path.join(out_dir, "CRISPRGeneEffect_clean.csv"),
            os.path.join(duplicate_dir, "crispr_duplicate_ModelID_rows.csv"),
            "crispr",
        )
    )

    # CNV wide matrix
    reports.append(
        clean_wide_matrix(
            os.path.join(raw_dir, FILES["cnv"]),
            os.path.join(out_dir, "PortalOmicsCNGeneLog2_clean.csv"),
            os.path.join(duplicate_dir, "cnv_duplicate_ModelID_rows.csv"),
            "cnv",
        )
    )

    report = {
        "version": "DepMap_Cleaning_v1",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_dir": raw_dir,
        "out_dir": out_dir,
        "policy": {
            "duplicate_policy": "Duplicate ModelID rows are logged. Cleaned matrix keeps the first row only.",
            "identifier_policy": "Unnamed: 0 is renamed to ModelID for wide matrices.",
            "filtering_policy": "No aggressive biological filtering during source cleaning.",
        },
        "files": reports,
    }

    report_path = os.path.join(out_dir, "depmap_cleaning_report_v1.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_depmap_clean()
