import os
import json
import hashlib
import re
from datetime import datetime

import pandas as pd


VERSION = "GDSC_DepMap_Mapping_v1"

GDSC_PASS_PATH = "outputs/training_ready/gdsc/gdsc_pass_training_grade_v1.csv"
DEPMAP_MODEL_PATH = "outputs/processed/depmap_ccle/Model_clean.csv"
OUT_DIR = "outputs/mapping/gdsc_depmap"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_name(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_depmap_lookup(model_df):
    model_id_col = first_existing_col(model_df, ["ModelID", "DepMap_ID", "DepMapID"])

    if model_id_col is None:
        raise ValueError("DepMap Model file has no ModelID/DepMap_ID column.")

    candidate_name_cols = [
        "CellLineName",
        "StrippedCellLineName",
        "CCLEName",
        "ModelName",
        "COSMICID",
        "SangerModelID",
    ]

    candidate_name_cols = [c for c in candidate_name_cols if c in model_df.columns]

    rows = []

    for _, row in model_df.iterrows():
        model_id = row.get(model_id_col, "")

        for col in candidate_name_cols:
            val = row.get(col, "")
            norm = normalize_name(val)

            if norm:
                rows.append({
                    "depmap_model_id": model_id,
                    "depmap_match_column": col,
                    "depmap_match_value": val,
                    "normalized_key": norm,
                })

    lookup = pd.DataFrame(rows)

    if len(lookup) == 0:
        raise ValueError("No usable DepMap model-name lookup keys generated.")

    key_counts = (
        lookup.groupby("normalized_key")["depmap_model_id"]
        .nunique()
        .reset_index(name="n_depmap_models")
    )

    lookup = lookup.merge(key_counts, on="normalized_key", how="left")
    lookup["depmap_key_status"] = lookup["n_depmap_models"].apply(
        lambda x: "AMBIGUOUS" if x > 1 else "UNIQUE"
    )

    return lookup


def run_gdsc_depmap_mapping(
    gdsc_path=GDSC_PASS_PATH,
    depmap_model_path=DEPMAP_MODEL_PATH,
    out_dir=OUT_DIR,
):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(gdsc_path):
        raise FileNotFoundError(f"Missing GDSC PASS dataset: {gdsc_path}")

    if not os.path.exists(depmap_model_path):
        raise FileNotFoundError(f"Missing DepMap Model_clean file: {depmap_model_path}")

    print("Loading GDSC PASS dataset...")
    gdsc = pd.read_csv(gdsc_path, low_memory=False)

    print("Loading DepMap Model metadata...")
    depmap_model = pd.read_csv(depmap_model_path, low_memory=False)

    gdsc_cell_col = first_existing_col(
        gdsc,
        ["cell_line_name", "model_name", "CellLineName", "CELL_LINE_NAME"]
    )

    gdsc_model_id_col = first_existing_col(
        gdsc,
        ["model_source_id", "COSMIC_ID", "cosmic_id", "SANGER_MODEL_ID"]
    )

    if gdsc_cell_col is None:
        raise ValueError("GDSC PASS dataset has no cell_line_name/model_name column.")

    depmap_lookup = build_depmap_lookup(depmap_model)

    gdsc["_gdsc_cell_key"] = gdsc[gdsc_cell_col].apply(normalize_name)

    mapped = gdsc.merge(
        depmap_lookup,
        left_on="_gdsc_cell_key",
        right_on="normalized_key",
        how="left",
    )

    mapped["gdsc_depmap_mapping_status"] = "UNMATCHED"

    mapped.loc[
        mapped["depmap_model_id"].notna() & (mapped["depmap_key_status"] == "UNIQUE"),
        "gdsc_depmap_mapping_status"
    ] = "MATCHED_UNIQUE"

    mapped.loc[
        mapped["depmap_model_id"].notna() & (mapped["depmap_key_status"] == "AMBIGUOUS"),
        "gdsc_depmap_mapping_status"
    ] = "REVIEW_AMBIGUOUS"

    mapped_training = mapped[
        mapped["gdsc_depmap_mapping_status"] == "MATCHED_UNIQUE"
    ].copy()

    unmatched = mapped[
        mapped["gdsc_depmap_mapping_status"] != "MATCHED_UNIQUE"
    ].copy()

    mapping_only_cols = [
        c for c in [
            "record_id",
            "drug_name_standard",
            gdsc_cell_col,
            gdsc_model_id_col,
            "_gdsc_cell_key",
            "depmap_model_id",
            "depmap_match_column",
            "depmap_match_value",
            "depmap_key_status",
            "gdsc_depmap_mapping_status",
        ]
        if c is not None and c in mapped.columns
    ]

    mapping_only = mapped[mapping_only_cols].copy()

    mapping_path = os.path.join(out_dir, "GDSC_DepMap_mapping_v1.csv")
    training_base_path = os.path.join(out_dir, "GDSC_DepMap_mapped_training_base_v1.csv")
    unmatched_path = os.path.join(out_dir, "GDSC_DepMap_unmatched_v1.csv")
    depmap_lookup_path = os.path.join(out_dir, "DepMap_model_lookup_v1.csv")
    summary_path = os.path.join(out_dir, "GDSC_DepMap_mapping_summary_v1.json")

    mapping_only.to_csv(mapping_path, index=False)
    mapped_training.to_csv(training_base_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)
    depmap_lookup.to_csv(depmap_lookup_path, index=False)

    total = len(gdsc)
    matched_unique = len(mapped_training)
    review_ambiguous = int((mapped["gdsc_depmap_mapping_status"] == "REVIEW_AMBIGUOUS").sum())
    unmatched_count = int((mapped["gdsc_depmap_mapping_status"] == "UNMATCHED").sum())

    summary = {
        "version": VERSION,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_files": {
            "gdsc_pass_path": gdsc_path,
            "gdsc_pass_sha256": sha256_file(gdsc_path),
            "depmap_model_path": depmap_model_path,
            "depmap_model_sha256": sha256_file(depmap_model_path),
        },
        "columns_used": {
            "gdsc_cell_col": gdsc_cell_col,
            "gdsc_model_id_col": gdsc_model_id_col,
        },
        "counts": {
            "gdsc_pass_rows": int(total),
            "mapped_unique_rows": int(matched_unique),
            "review_ambiguous_rows": int(review_ambiguous),
            "unmatched_rows": int(unmatched_count),
            "mapped_unique_percent": round((matched_unique / total) * 100, 2) if total else 0,
            "unmatched_percent": round((unmatched_count / total) * 100, 2) if total else 0,
        },
        "policy": {
            "training_mapping_policy": "Only MATCHED_UNIQUE rows are allowed into mapped training base.",
            "ambiguous_policy": "Ambiguous DepMap matches are excluded and written to unmatched/review file.",
            "unmatched_policy": "Unmatched rows are excluded from mapped training base.",
            "normalization_policy": "Cell-line names are uppercased and stripped of non-alphanumeric characters before matching.",
        },
        "outputs": {
            "mapping": mapping_path,
            "mapped_training_base": training_base_path,
            "unmatched": unmatched_path,
            "depmap_lookup": depmap_lookup_path,
            "summary": summary_path,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_gdsc_depmap_mapping()
