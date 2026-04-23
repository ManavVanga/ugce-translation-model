import pandas as pd
import hashlib


DEDUP_KEY_COLUMNS = [
    "drug_name_standard",
    "vitro_system_class",
    "tissue_context",
    "disease_context",
    "cell_line_name",
    "assay_type",
    "assay_endpoint",
    "dose_normalized_uM",
    "exposure_time_hours",
    "response_value_standard",
    "response_metric_standard",
    "invivo_species",
    "invivo_outcome_label",
    "human_outcome_label",
]


def clean_for_key(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def build_dedup_key(row):
    values = [clean_for_key(row.get(col, "")) for col in DEDUP_KEY_COLUMNS]
    joined = "||".join(values)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def add_dedup_keys(df):
    df = df.copy()
    df["dedup_key"] = df.apply(build_dedup_key, axis=1)
    return df


def deduplicate_against_existing(new_df, existing_df):
    new_df = add_dedup_keys(new_df)

    if existing_df is None or len(existing_df) == 0:
        new_df["is_duplicate"] = False
        return new_df

    existing_df = add_dedup_keys(existing_df)
    existing_keys = set(existing_df["dedup_key"].astype(str))

    new_df["is_duplicate"] = new_df["dedup_key"].astype(str).isin(existing_keys)
    return new_df


def split_new_vs_duplicate(df):
    new_only = df[df["is_duplicate"] == False].copy()
    duplicates = df[df["is_duplicate"] == True].copy()
    return new_only, duplicates
