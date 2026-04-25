import os
import json
import hashlib
from datetime import datetime

import pandas as pd
import numpy as np


RAW_PATH = "data_sources/gdsc/raw/gdsc_drug_response.csv"
OUT_DIR = "outputs/processed/gdsc"
OUT_PATH = os.path.join(OUT_DIR, "gdsc_iviv_normalized_v1.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "gdsc_normalization_summary_v1.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_col(df, candidates):
    lower_map = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        if name.lower().strip() in lower_map:
            return lower_map[name.lower().strip()]
    return None


def normalize_gdsc(raw_path=RAW_PATH, out_path=OUT_PATH):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing GDSC raw file: {raw_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pd.read_csv(raw_path, low_memory=False)

    drug_col = find_col(df, [
        "drug_name", "DRUG_NAME", "compound_name", "Compound", "drug", "Drug Name"
    ])

    cell_col = find_col(df, [
        "cell_line_name", "CELL_LINE_NAME", "Cell line name", "cell_line",
        "COSMIC_ID", "model_name"
    ])

    tissue_col = find_col(df, [
        "tissue", "TISSUE", "TCGA_DESC", "cancer_type", "lineage"
    ])

    ic50_col = find_col(df, [
        "LN_IC50", "ln_ic50", "IC50", "ic50", "IC50_uM", "response_value"
    ])

    auc_col = find_col(df, [
        "AUC", "auc"
    ])

    drug_id_col = find_col(df, [
        "DRUG_ID", "drug_id", "Compound ID", "compound_id"
    ])

    cell_id_col = find_col(df, [
        "COSMIC_ID", "cosmic_id", "SANGER_MODEL_ID", "model_id"
    ])

    if drug_col is None:
        raise ValueError("Could not detect drug column in GDSC raw file.")

    if cell_col is None:
        raise ValueError("Could not detect cell line/model column in GDSC raw file.")

    if ic50_col is None and auc_col is None:
        raise ValueError("Could not detect IC50/LN_IC50/AUC response column in GDSC raw file.")

    response_col = ic50_col if ic50_col is not None else auc_col

    if response_col.lower() in {"ln_ic50", "ln ic50"} or response_col == "LN_IC50":
        response_metric = "LN_IC50"
        assay_endpoint = "LN_IC50"
    elif response_col.lower() in {"auc"}:
        response_metric = "AUC"
        assay_endpoint = "AUC"
    else:
        response_metric = "IC50_uM"
        assay_endpoint = "IC50"

    out = pd.DataFrame()

    out["record_id"] = [f"GDSC_{i+1:09d}" for i in range(len(df))]
    out["study_id"] = "GDSC"
    out["source_type"] = "database"
    out["source_name"] = "GDSC"
    out["source_database"] = "GDSC"
    out["source_url"] = ""
    out["data_added_date"] = datetime.now().strftime("%Y-%m-%d")
    out["curated_by"] = "gdsc_normalization_v1"
    out["review_status"] = "auto_normalized"

    out["drug_name_standard"] = df[drug_col].astype(str).str.strip()
    out["drug_source_id"] = df[drug_id_col].astype(str).str.strip() if drug_id_col else ""

    out["model_name"] = df[cell_col].astype(str).str.strip()
    out["cell_line_name"] = df[cell_col].astype(str).str.strip()
    out["model_source_id"] = df[cell_id_col].astype(str).str.strip() if cell_id_col else ""

    out["vitro_system_class"] = "cell_line"
    out["tissue_context"] = df[tissue_col].astype(str).str.strip() if tissue_col else ""
    out["species"] = "human"
    out["disease_context"] = out["tissue_context"]

    out["intervention_type"] = "small_molecule_drug"
    out["assay_type"] = "drug_sensitivity"
    out["assay_endpoint"] = assay_endpoint

    out["dose_normalized_uM"] = np.nan
    out["exposure_time_hours"] = np.nan

    out["response_value_standard"] = pd.to_numeric(df[response_col], errors="coerce")
    out["response_metric_standard"] = response_metric

    out["effect_direction"] = ""
    out["outcome_level"] = "in_vitro"

    out["invivo_outcome_label"] = ""
    out["human_outcome_label"] = ""
    out["evidence_weight"] = ""
    out["evidence_sentence"] = ""
    out["evidence_window"] = ""

    out["raw_response_column"] = response_col
    out["raw_drug_column"] = drug_col
    out["raw_model_column"] = cell_col

    out.to_csv(out_path, index=False)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "GDSC_NORMALIZATION_COMPLETE",
        "raw_path": raw_path,
        "raw_sha256": sha256_file(raw_path),
        "output_path": out_path,
        "raw_rows": int(len(df)),
        "normalized_rows": int(len(out)),
        "detected_columns": {
            "drug_col": drug_col,
            "drug_id_col": drug_id_col,
            "cell_col": cell_col,
            "cell_id_col": cell_id_col,
            "tissue_col": tissue_col,
            "response_col": response_col,
            "response_metric_standard": response_metric,
        },
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return out


if __name__ == "__main__":
    normalize_gdsc()
