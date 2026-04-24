import os
import re
import json
import pandas as pd
from datetime import datetime
from src.utils.config import get_base_output_dir


RAW_DIR = "data_sources/gdsc/raw"
OUT_SUBDIR = "source_normalized"


POSSIBLE_RESPONSE_FILES = [
    "gdsc_drug_response.csv",
    "gdsc_response.csv",
    "gdsc_ic50.csv",
    "gdsc_drug_sensitivity.csv",
]

POSSIBLE_CELL_FILES = [
    "gdsc_cell_lines.csv",
    "gdsc_cellline_metadata.csv",
    "gdsc_models.csv",
]

POSSIBLE_DRUG_FILES = [
    "gdsc_compounds.csv",
    "gdsc_drugs.csv",
    "gdsc_drug_metadata.csv",
]


def clean(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def find_first_existing(folder, candidates):
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def find_col(df, options):
    lower_map = {c.lower().strip(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lower_map:
            return lower_map[opt.lower()]
    for c in df.columns:
        low = c.lower()
        for opt in options:
            if opt.lower() in low:
                return c
    return None


def safe_value(row, col):
    if col is None:
        return ""
    return clean(row.get(col, ""))


def safe_float(row, col):
    if col is None:
        return None
    val = row.get(col, None)
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def load_optional_csv(path):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def build_lookup(df, key_options):
    if df is None or len(df) == 0:
        return {}, None

    key_col = find_col(df, key_options)
    if key_col is None:
        return {}, None

    lookup = {}
    for _, r in df.iterrows():
        key = clean(r.get(key_col, ""))
        if key:
            lookup[key] = r.to_dict()

    return lookup, key_col


def normalize_gdsc_rows(response_df, cell_df, drug_df):
    rows = []

    cell_lookup, cell_key_col = build_lookup(
        cell_df,
        ["cell_line_id", "cosmic_id", "COSMIC_ID", "model_id", "CELL_LINE_ID"]
    )

    drug_lookup, drug_key_col = build_lookup(
        drug_df,
        ["drug_id", "DRUG_ID", "compound_id", "drug_name", "DRUG_NAME"]
    )

    col_cell_id = find_col(response_df, ["cell_line_id", "COSMIC_ID", "cosmic_id", "model_id"])
    col_cell_name = find_col(response_df, ["cell_line_name", "CELL_LINE_NAME", "model_name"])
    col_drug_id = find_col(response_df, ["drug_id", "DRUG_ID", "compound_id"])
    col_drug_name = find_col(response_df, ["drug_name", "DRUG_NAME", "compound_name", "screening_site"])

    col_ic50 = find_col(response_df, ["ic50", "ln_ic50", "IC50", "LN_IC50", "z_score"])
    col_auc = find_col(response_df, ["auc", "AUC", "activity_area"])
    col_tissue = find_col(response_df, ["tissue", "lineage", "cancer_type", "TCGA_DESC", "GDSC_TISSUE"])
    col_dataset = find_col(response_df, ["dataset", "source", "DATASET"])

    for idx, r in response_df.iterrows():
        cell_id = safe_value(r, col_cell_id)
        drug_id = safe_value(r, col_drug_id)

        cell_meta = cell_lookup.get(cell_id, {}) if cell_id else {}
        drug_meta = drug_lookup.get(drug_id, {}) if drug_id else {}

        cell_line_name = safe_value(r, col_cell_name)
        if not cell_line_name:
            for k in ["cell_line_name", "CELL_LINE_NAME", "model_name", "Model Name"]:
                if k in cell_meta:
                    cell_line_name = clean(cell_meta.get(k))
                    break

        drug_name = safe_value(r, col_drug_name)
        if not drug_name:
            for k in ["drug_name", "DRUG_NAME", "compound_name", "Compound Name"]:
                if k in drug_meta:
                    drug_name = clean(drug_meta.get(k))
                    break

        tissue = safe_value(r, col_tissue)
        if not tissue:
            for k in ["tissue", "lineage", "cancer_type", "TCGA_DESC", "GDSC_TISSUE"]:
                if k in cell_meta:
                    tissue = clean(cell_meta.get(k))
                    break

        ic50_val = safe_float(r, col_ic50)
        auc_val = safe_float(r, col_auc)

        endpoint_type = ""
        response_value = None
        response_metric = ""

        if ic50_val is not None:
            endpoint_type = "IC50"
            response_value = ic50_val
            response_metric = "GDSC_reported_IC50_or_lnIC50"
        elif auc_val is not None:
            endpoint_type = "AUC"
            response_value = auc_val
            response_metric = "GDSC_reported_AUC"
        else:
            continue

        row = {
            "record_id": f"GDSC_{idx+1:09d}",
            "study_id": "GDSC",
            "source_type": "database",
            "source_name": "GDSC",
            "source_url": "",
            "source_database": "GDSC",
            "data_added_date": datetime.now().strftime("%Y-%m-%d"),
            "curated_by": "gdsc_intake_v1",
            "review_status": "auto_normalized",

            "disease_context": clean(tissue),
            "tissue_context": clean(tissue),
            "indication_class": "oncology",
            "genetic_context": "",

            "model_type": "in_vitro",
            "model_subtype": "2d_cell_line",
            "vitro_system_class": "2d_cell_line",
            "species": "human",
            "cell_line_name": cell_line_name,
            "model_name": cell_line_name,
            "model_identifier": cell_id,
            "model_complexity_level": "low_2d",

            "exposure_type": "drug",
            "drug_name_standard": drug_name,
            "drug_identifier": drug_id,
            "drug_class": "",
            "mechanism_of_action": "",
            "compound_id_pubchem": "",
            "compound_id_chembl": "",

            "dose_value": "",
            "dose_unit": "",
            "dose_normalized_uM": "",
            "route_of_administration": "",
            "exposure_time_hours": "",
            "treatment_schedule": "",

            "endpoint_type": endpoint_type,
            "endpoint_category": "phenotypic_drug_response",
            "endpoint_name": endpoint_type,
            "measurement_unit": response_metric,
            "response_value": response_value,
            "response_value_standard": response_value,
            "response_metric_standard": response_metric,
            "response_direction": "",
            "effect_direction": "",

            "omics_type": "",
            "gene_symbol": "",
            "pathway": "",
            "fold_change": "",
            "p_value": "",
            "biomarker_flag": "",

            "animal_model_type": "",
            "tumor_response": "",
            "survival_outcome": "",
            "toxicity_signal": "",
            "pk_parameters": "",

            "clinical_phase": "",
            "patient_population": "",
            "clinical_endpoint": "",
            "clinical_response_value": "",
            "clinical_response_class": "",
            "adverse_event_type": "",

            "linked_invitro_id": f"GDSC_{cell_id}_{drug_id}",
            "linked_invivo_id": "",
            "linked_clinical_id": "",
            "translation_pathway": "in_vitro_to_human_target_future",
            "translation_consistency": "unknown",

            "translation_label": "",
            "translation_confidence_score": "",
            "species_relevance_score": 0.5,
            "model_predictivity_score": "",
            "evidence_weight": "structured_database",

            "evidence_sentence": "",
            "evidence_table": "GDSC drug response table",
            "figure_reference": "",
            "section_source": "structured_database",

            "qc_status": "UNREVIEWED",
            "qc_flags": "",
            "data_completeness_score": "",
            "audit_version": "gdsc_intake_v1",
        }

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    base_dir = get_base_output_dir()

    raw_dir = os.path.join(RAW_DIR)
    output_dir = os.path.join(base_dir, OUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    response_path = find_first_existing(raw_dir, POSSIBLE_RESPONSE_FILES)
    cell_path = find_first_existing(raw_dir, POSSIBLE_CELL_FILES)
    drug_path = find_first_existing(raw_dir, POSSIBLE_DRUG_FILES)

    print("GDSC raw dir:", raw_dir)
    print("Response file:", response_path)
    print("Cell metadata file:", cell_path)
    print("Drug metadata file:", drug_path)

    if response_path is None:
        raise FileNotFoundError(
            "No GDSC response file found. Put one of these in data_sources/gdsc/raw/: "
            + ", ".join(POSSIBLE_RESPONSE_FILES)
        )

    response_df = pd.read_csv(response_path)
    cell_df = load_optional_csv(cell_path)
    drug_df = load_optional_csv(drug_path)

    print("Response shape:", response_df.shape)
    print("Cell metadata shape:", cell_df.shape)
    print("Drug metadata shape:", drug_df.shape)

    normalized_df = normalize_gdsc_rows(response_df, cell_df, drug_df)

    out_path = os.path.join(output_dir, "gdsc_iviv_rows_v1.csv")
    normalized_df.to_csv(out_path, index=False)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "GDSC",
        "response_file": response_path,
        "cell_file": cell_path,
        "drug_file": drug_path,
        "input_rows": int(len(response_df)),
        "normalized_rows": int(len(normalized_df)),
        "output_file": out_path,
    }

    summary_path = os.path.join(output_dir, "gdsc_intake_summary_v1.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_path)
    print("Saved summary:", summary_path)
    print(summary)

    if len(normalized_df) > 0:
        print(normalized_df.head(10).to_string(index=False))
