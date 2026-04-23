import re
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# DRUG NORMALIZATION
# ------------------------------------------------------------
DRUG_SYNONYMS = {
    "acetaminophen": ["paracetamol", "acetaminophen"],
    "cisplatin": ["cisplatin"],
    "doxorubicin": ["doxorubicin", "adriamycin"],
    "imatinib": ["imatinib", "gleevec"],
    "lenvatinib": ["lenvatinib"],
    "paclitaxel": ["paclitaxel", "taxol"],
    "sorafenib": ["sorafenib", "nexavar"],
    "tamoxifen": ["tamoxifen"],
    "berberine": ["berberine"],
}

PUBCHEM_IDS = {
    "acetaminophen": "1983",
    "cisplatin": "441203",
    "doxorubicin": "31703",
    "imatinib": "5291",
    "lenvatinib": "9823820",
    "paclitaxel": "36314",
    "sorafenib": "216239",
    "tamoxifen": "2733526",
    "berberine": "2353",
}

CHEMBL_IDS = {
    "acetaminophen": "CHEMBL112",
    "cisplatin": "CHEMBL1259",
    "doxorubicin": "CHEMBL53463",
    "imatinib": "CHEMBL941",
    "lenvatinib": "CHEMBL1289601",
    "paclitaxel": "CHEMBL428647",
    "sorafenib": "CHEMBL1336",
    "tamoxifen": "CHEMBL83",
    "berberine": "",
}

# ------------------------------------------------------------
# MODEL NORMALIZATION
# ------------------------------------------------------------
MODEL_NORMALIZATION = {
    "organoid": "organoid",
    "organoids": "organoid",
    "organ_on_chip": "organ_on_chip",
    "organ-on-chip": "organ_on_chip",
    "mps": "mps",
    "microphysiological": "mps",
    "2d": "2d",
    "2d culture": "2d",
    "3d_spheroid": "3d_spheroid",
    "spheroid": "3d_spheroid",
    "crisper_or_genetic": "crisper_or_genetic",
    "crispr": "crisper_or_genetic",
}

# ------------------------------------------------------------
# ASSAY / ENDPOINT NORMALIZATION
# ------------------------------------------------------------
ASSAY_NORMALIZATION = {
    "viability": "viability",
    "cell viability": "viability",
    "toxicity": "toxicity",
    "cytotoxicity": "toxicity",
    "screening": "screening",
    "apoptosis": "apoptosis",
    "proliferation": "proliferation",
}

ENDPOINT_NORMALIZATION = {
    "ic50": "IC50",
    "ic50_um": "IC50_uM",
    "viability_percent": "viability_percent",
}

# ------------------------------------------------------------
# OUTCOME NORMALIZATION
# ------------------------------------------------------------
OUTCOME_NORMALIZATION = {
    "tumor_suppression": "tumor_response_preclinical",
    "partial_response": "partial_response",
    "hepatotoxicity": "hepatotoxicity",
    "nephrotoxicity": "nephrotoxicity",
    "cardiotoxicity": "cardiotoxicity",
    "toxicity": "general_toxicity",
    "efficacy": "general_efficacy",
}

# Optional mapping to broad human-friendly domains
OUTCOME_DOMAIN_MAP = {
    "tumor_response_preclinical": "efficacy",
    "partial_response": "efficacy",
    "hepatotoxicity": "toxicity",
    "nephrotoxicity": "toxicity",
    "cardiotoxicity": "toxicity",
    "general_toxicity": "toxicity",
    "general_efficacy": "efficacy",
}

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()

def normalize_key(x):
    x = clean_text(x).lower()
    x = re.sub(r"[\(\)\[\]\{\},;:/\\\-\+]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def first_canonical_from_synonyms(raw_name, synonym_dict):
    raw = normalize_key(raw_name)
    if raw == "":
        return ""
    for canonical, synonyms in synonym_dict.items():
        for s in synonyms:
            if normalize_key(s) == raw:
                return canonical
    return raw_name.strip().lower() if raw_name else ""

def normalize_lookup(raw_value, mapping):
    key = normalize_key(raw_value).replace(" ", "_")
    if key in mapping:
        return mapping[key]

    key2 = normalize_key(raw_value)
    if key2 in mapping:
        return mapping[key2]

    # fallback: raw stripped
    return clean_text(raw_value)

def to_float_or_nan(x):
    try:
        if clean_text(x) == "":
            return np.nan
        return float(x)
    except:
        return np.nan

# ------------------------------------------------------------
# MAIN NORMALIZER
# ------------------------------------------------------------
def normalize_translation_rows(df):
    df = df.copy()

    # drug normalization
    df["drug_name_standard"] = df["drug_name_standard"].apply(
        lambda x: first_canonical_from_synonyms(x, DRUG_SYNONYMS)
    )

    # IDs
    df["compound_id_pubchem"] = df["drug_name_standard"].apply(
        lambda x: PUBCHEM_IDS.get(clean_text(x).lower(), "")
    )
    df["compound_id_chembl"] = df["drug_name_standard"].apply(
        lambda x: CHEMBL_IDS.get(clean_text(x).lower(), "")
    )

    # model type
    df["vitro_system_class"] = df["vitro_system_class"].apply(
        lambda x: normalize_lookup(x, MODEL_NORMALIZATION)
    )

    # assay type
    df["assay_type"] = df["assay_type"].apply(
        lambda x: normalize_lookup(x, ASSAY_NORMALIZATION)
    )

    # endpoint
    df["assay_endpoint"] = df["assay_endpoint"].apply(
        lambda x: normalize_lookup(x, ENDPOINT_NORMALIZATION)
    )

    # response metric
    df["response_metric_standard"] = df["response_metric_standard"].apply(
        lambda x: normalize_lookup(x, ENDPOINT_NORMALIZATION)
    )

    # outcome labels
    df["invivo_outcome_label"] = df["invivo_outcome_label"].apply(
        lambda x: normalize_lookup(x, OUTCOME_NORMALIZATION)
    )
    df["human_outcome_label"] = df["human_outcome_label"].apply(
        lambda x: normalize_lookup(x, OUTCOME_NORMALIZATION)
    )

    # if human outcome missing, keep animal; if animal missing, keep human
    df["normalized_outcome_domain"] = df.apply(
        lambda r: OUTCOME_DOMAIN_MAP.get(
            clean_text(r["human_outcome_label"]) or clean_text(r["invivo_outcome_label"]),
            ""
        ),
        axis=1
    )

    # numeric cleanup
    numeric_cols = [
        "dose_normalized_uM",
        "exposure_time_hours",
        "response_value_standard",
        "replicate_count",
        "species_relevance_score",
        "translation_confidence_score"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(to_float_or_nan)

    # intervention type normalization
    df["intervention_type"] = df["intervention_type"].apply(
        lambda x: "small_molecule" if clean_text(x).lower() in ["small_molecule", "small molecule"] else clean_text(x)
    )

    # evidence weight normalization
    df["evidence_weight"] = df["evidence_weight"].apply(
        lambda x: clean_text(x).lower()
    )

    # effect direction normalization
    df["effect_direction"] = df["effect_direction"].apply(
        lambda x: clean_text(x).lower()
    )

    # fill blanks safely
    fill_text_cols = [
        "compound_id_pubchem", "compound_id_chembl", "smiles",
        "cell_line_name", "disease_context", "invivo_species",
        "invivo_outcome_label", "human_outcome_label",
        "normalized_outcome_domain"
    ]
    for col in fill_text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df
