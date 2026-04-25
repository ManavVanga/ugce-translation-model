import os
import json
import hashlib
from datetime import datetime

import pandas as pd
import numpy as np


QC_VERSION = "DepMap_QC_v1"


RAW_DIR = "data_sources/depmap_ccle/raw"
OUT_DIR = "outputs/qc/depmap_ccle/DepMap_QC_v1"


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


def is_missing(x):
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def read_csv_safe(path, nrows=None):
    return pd.read_csv(path, low_memory=False, nrows=nrows)


def detect_identifier_column(df):
    candidates = [
        "ModelID",
        "Model_ID",
        "DepMap_ID",
        "DepMapID",
        "model_id",
        "depmap_id",
        "ACHILLES_ID",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def qc_model_file(path):
    df = read_csv_safe(path)

    required_any_id = detect_identifier_column(df)

    possible_name_cols = [
        "ModelID",
        "ModelName",
        "CellLineName",
        "StrippedCellLineName",
        "CCLEName",
        "OncotreeLineage",
        "OncotreePrimaryDisease",
    ]

    present_name_cols = [c for c in possible_name_cols if c in df.columns]

    flags = []

    if required_any_id is None:
        flags.append("MISSING_MODEL_IDENTIFIER_COLUMN")

    if len(present_name_cols) == 0:
        flags.append("MISSING_MODEL_NAME_OR_LINEAGE_COLUMNS")

    duplicate_model_ids = 0
    missing_model_ids = 0

    if required_any_id is not None:
        missing_model_ids = int(df[required_any_id].apply(is_missing).sum())
        duplicate_model_ids = int(df.duplicated(subset=[required_any_id], keep=False).sum())

        if missing_model_ids > 0:
            flags.append("MISSING_MODEL_IDS")

        if duplicate_model_ids > 0:
            flags.append("DUPLICATE_MODEL_IDS")

    summary = {
        "file_type": "model",
        "path": path,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "sha256": sha256_file(path),
        "identifier_column": required_any_id,
        "present_mapping_columns": present_name_cols,
        "missing_model_ids": missing_model_ids,
        "duplicate_model_ids": duplicate_model_ids,
        "flags": flags,
        "status": "PASS" if len(flags) == 0 else "REVIEW",
    }

    return summary


def qc_wide_matrix_file(path, file_type):
    df_head = read_csv_safe(path, nrows=5)
    id_col = detect_identifier_column(df_head)

    flags = []

    if id_col is None:
        first_col = df_head.columns[0] if len(df_head.columns) else None
        id_col = first_col
        flags.append("IDENTIFIER_COLUMN_INFERRED_FROM_FIRST_COLUMN")

    df = read_csv_safe(path)

    rows = len(df)
    cols = df.shape[1]

    missing_ids = 0
    duplicate_ids = 0

    if id_col in df.columns:
        missing_ids = int(df[id_col].apply(is_missing).sum())
        duplicate_ids = int(df.duplicated(subset=[id_col], keep=False).sum())

        if missing_ids > 0:
            flags.append("MISSING_ROW_IDENTIFIERS")

        if duplicate_ids > 0:
            flags.append("DUPLICATE_ROW_IDENTIFIERS")
    else:
        flags.append("IDENTIFIER_COLUMN_NOT_FOUND_AFTER_FULL_READ")

    numeric_cols = [c for c in df.columns if c != id_col]
    numeric_sample = df[numeric_cols].select_dtypes(include=[np.number])

    numeric_column_count = int(numeric_sample.shape[1])
    total_feature_columns = int(len(numeric_cols))

    if total_feature_columns == 0:
        flags.append("NO_FEATURE_COLUMNS")

    if numeric_column_count == 0:
        flags.append("NO_NUMERIC_FEATURE_COLUMNS")

    missing_fraction = None
    if len(numeric_cols) > 0:
        missing_fraction = float(df[numeric_cols].isna().sum().sum() / (rows * len(numeric_cols)))

    all_zero_rows = 0
    if numeric_column_count > 0:
        all_zero_rows = int((numeric_sample.fillna(0).abs().sum(axis=1) == 0).sum())
        if all_zero_rows == rows:
            flags.append("ALL_ROWS_ZERO_NUMERIC_VALUES")

    # Structural QC only: do not fail aggressively unless file is unusable.
    hard_fail_flags = {
        "NO_FEATURE_COLUMNS",
        "NO_NUMERIC_FEATURE_COLUMNS",
        "IDENTIFIER_COLUMN_NOT_FOUND_AFTER_FULL_READ",
        "ALL_ROWS_ZERO_NUMERIC_VALUES",
    }

    status = "FAIL" if any(f in hard_fail_flags for f in flags) else ("REVIEW" if flags else "PASS")

    summary = {
        "file_type": file_type,
        "path": path,
        "rows": int(rows),
        "columns": int(cols),
        "sha256": sha256_file(path),
        "identifier_column": id_col,
        "feature_columns": total_feature_columns,
        "numeric_feature_columns_detected": numeric_column_count,
        "missing_identifier_rows": missing_ids,
        "duplicate_identifier_rows": duplicate_ids,
        "missing_fraction_all_features": missing_fraction,
        "all_zero_rows_numeric_subset": all_zero_rows,
        "flags": flags,
        "status": status,
    }

    return summary


def qc_mutation_file(path):
    df = read_csv_safe(path)

    flags = []

    id_col = detect_identifier_column(df)

    gene_candidates = [
        "HugoSymbol",
        "Hugo_Symbol",
        "Gene",
        "gene",
        "EntrezGeneId",
    ]
    variant_candidates = [
        "VariantInfo",
        "Variant_Classification",
        "ProteinChange",
        "OncotatorVariantClassification",
        "MutationType",
    ]

    gene_cols = [c for c in gene_candidates if c in df.columns]
    variant_cols = [c for c in variant_candidates if c in df.columns]

    if id_col is None:
        flags.append("MISSING_MODEL_IDENTIFIER_COLUMN")

    if len(gene_cols) == 0:
        flags.append("MISSING_GENE_COLUMN")

    if len(variant_cols) == 0:
        flags.append("MISSING_VARIANT_ANNOTATION_COLUMN")

    missing_ids = 0
    if id_col is not None:
        missing_ids = int(df[id_col].apply(is_missing).sum())
        if missing_ids > 0:
            flags.append("MISSING_MODEL_IDS")

    missing_gene_rows = 0
    if len(gene_cols) > 0:
        main_gene_col = gene_cols[0]
        missing_gene_rows = int(df[main_gene_col].apply(is_missing).sum())
        if missing_gene_rows > 0:
            flags.append("MISSING_GENE_VALUES")

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        flags.append("EXACT_DUPLICATE_MUTATION_ROWS")

    hard_fail_flags = {
        "MISSING_MODEL_IDENTIFIER_COLUMN",
        "MISSING_GENE_COLUMN",
    }

    status = "FAIL" if any(f in hard_fail_flags for f in flags) else ("REVIEW" if flags else "PASS")

    summary = {
        "file_type": "mutations",
        "path": path,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "sha256": sha256_file(path),
        "identifier_column": id_col,
        "gene_columns_detected": gene_cols,
        "variant_columns_detected": variant_cols,
        "missing_model_id_rows": missing_ids,
        "missing_gene_rows": missing_gene_rows,
        "exact_duplicate_rows": duplicate_rows,
        "flags": flags,
        "status": status,
    }

    return summary


def run_depmap_qc(raw_dir=RAW_DIR, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    summaries = []
    missing_files = []

    for file_type, filename in FILES.items():
        path = os.path.join(raw_dir, filename)

        if not os.path.exists(path):
            missing_files.append(filename)
            summaries.append({
                "file_type": file_type,
                "path": path,
                "status": "FAIL",
                "flags": ["MISSING_FILE"],
            })
            continue

        print(f"Running QC for {file_type}: {path}")

        if file_type == "model":
            summaries.append(qc_model_file(path))
        elif file_type == "mutations":
            summaries.append(qc_mutation_file(path))
        else:
            summaries.append(qc_wide_matrix_file(path, file_type))

    summary_df = pd.DataFrame(summaries)
    summary_csv = os.path.join(out_dir, "depmap_qc_file_summary_v1.csv")
    summary_json = os.path.join(out_dir, "depmap_qc_summary_v1.json")

    summary_df.to_csv(summary_csv, index=False)

    status_counts = summary_df["status"].value_counts(dropna=False).to_dict()

    overall_status = "PASS"
    if "FAIL" in status_counts:
        overall_status = "FAIL"
    elif "REVIEW" in status_counts:
        overall_status = "REVIEW"

    report = {
        "qc_version": QC_VERSION,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_dir": raw_dir,
        "out_dir": out_dir,
        "expected_files": FILES,
        "missing_files": missing_files,
        "file_status_counts": status_counts,
        "overall_status": overall_status,
        "policy": {
            "qc_type": "structural_source_qc",
            "filtering_policy": "Do not aggressively drop DepMap rows during source QC. Filtering occurs during GDSC-DepMap mapping.",
            "mandatory_file": "Model.csv",
            "mapping_key_priority": "ModelID / DepMap_ID / first identifier column depending on file structure.",
            "hard_fail_conditions": [
                "missing file",
                "missing model identifier in key files",
                "missing gene column in mutation file",
                "no numeric feature columns in wide matrix files",
            ],
        },
        "outputs": {
            "summary_csv": summary_csv,
            "summary_json": summary_json,
        },
        "files": summaries,
    }

    with open(summary_json, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_depmap_qc()
