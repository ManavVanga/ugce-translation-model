import re
import pandas as pd
import numpy as np


DRUG_PATTERNS = [
    r"\b(berberine|cisplatin|sorafenib|lenvatinib|paclitaxel|doxorubicin|imatinib|acetaminophen|tamoxifen)\b",
    r"\b([A-Z][a-z]{2,}(?:mab|nib|platin|rubicin|fenib|tinib|parib|setron|mycin|taxel|ciclib|tidine|cycline))\b",
]

CELL_LINE_PATTERNS = [
    r"\b(Hep3B|HepG2|MCF7|A549|K562|HCT116|HT29|U2OS|HeLa|293T)\b"
]

TISSUE_RULES = [
    ("liver", ["liver", "hepatic", "hepatocyte", "hep3b", "hepg2"]),
    ("colon", ["colon", "colorectal", "intestinal", "hct116", "ht29"]),
    ("breast", ["breast", "mcf7"]),
    ("lung", ["lung", "a549"]),
    ("brain", ["brain", "neuronal", "neuron", "glioblastoma"]),
    ("kidney", ["kidney", "renal"]),
    ("prostate", ["prostate"]),
    ("stomach", ["gastric"]),
    ("testis", ["testicular", "testis"]),
]

DISEASE_RULES = [
    ("HCC", ["hcc", "hepatocellular carcinoma", "liver cancer"]),
    ("CRC", ["colorectal cancer", "crc", "colon cancer"]),
    ("breast_cancer", ["breast cancer", "mcf7"]),
    ("lung_cancer", ["lung cancer", "nsclc", "a549"]),
    ("gbm", ["glioblastoma", "gbm"]),
    ("healthy", ["healthy", "normal tissue"]),
]

MODEL_RULES = [
    ("organoid", ["organoid", "organoids"]),
    ("organ_on_chip", ["organ-on-chip", "organ on chip", "chip model"]),
    ("mps", ["microphysiological", "mps"]),
    ("crisper_or_genetic", ["crispr", "knockout", "knock-down", "knockdown", "gene editing"]),
    ("2d", ["2d culture", "cell line", "monolayer"]),
    ("3d_spheroid", ["spheroid", "3d culture", "three-dimensional"]),
]

ASSAY_RULES = [
    ("viability", ["viability", "cell viability", "survival"]),
    ("toxicity", ["toxicity", "cytotoxicity", "hepatotoxicity"]),
    ("apoptosis", ["apoptosis"]),
    ("proliferation", ["proliferation", "growth inhibition"]),
    ("screening", ["screening", "drug-response profiling", "drug response profiling"]),
]

OUTCOME_RULES = [
    ("tumor_suppression", ["tumor suppression", "tumour suppression", "reduced tumor growth", "xenograft inhibition"]),
    ("partial_response", ["partial response", "response rate", "clinical response"]),
    ("hepatotoxicity", ["hepatotoxicity", "drug induced liver injury", "dili", "liver injury"]),
    ("nephrotoxicity", ["nephrotoxicity", "kidney injury"]),
    ("cardiotoxicity", ["cardiotoxicity", "qt prolongation"]),
    ("toxicity", ["toxicity", "adverse event", "adverse effects"]),
    ("efficacy", ["efficacy", "therapeutic effect", "anti-tumor activity"]),
]


def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()

def find_first_pattern(text, patterns):
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            return m.group(1)
    return ""

def find_rule_label(text, rules):
    for label, keywords in rules:
        for kw in keywords:
            if kw.lower() in text:
                return label
    return ""

def normalize_conc_to_uM(value, unit):
    try:
        value = float(value)
    except:
        return np.nan

    unit = str(unit).strip()
    if unit in ["uM", "µM"]:
        return value
    if unit == "nM":
        return value / 1000.0
    if unit == "mM":
        return value * 1000.0
    return np.nan

def normalize_time_to_hours(value, unit):
    try:
        value = float(value)
    except:
        return np.nan

    unit = str(unit).lower().strip()
    if unit in ["h", "hr", "hrs", "hour", "hours"]:
        return value
    if unit in ["day", "days"]:
        return value * 24.0
    return np.nan

def extract_ic50(text):
    m = re.search(r"\bIC50\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(nM|uM|µM|mM)", text, re.I)
    if m:
        raw_value = float(m.group(1))
        raw_unit = m.group(2)
        value_uM = normalize_conc_to_uM(raw_value, raw_unit)
        return value_uM, raw_unit
    return np.nan, ""

def extract_general_dose(text):
    m = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(nM|uM|µM|mM)\b", text, re.I)
    if m:
        raw_value = float(m.group(1))
        raw_unit = m.group(2)
        value_uM = normalize_conc_to_uM(raw_value, raw_unit)
        return value_uM, raw_unit
    return np.nan, ""

def extract_exposure_time(text):
    m = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|day|days)\b", text, re.I)
    if m:
        return normalize_time_to_hours(m.group(1), m.group(2))
    return np.nan

def extract_viability_percent(text):
    m = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cell\s+)?viability\b", text, re.I)
    if m:
        return float(m.group(1))
    return np.nan

def extract_replicate_count(text):
    m = re.search(r"\bn\s*=\s*([0-9]+)\b", text, re.I)
    if m:
        return int(m.group(1))
    return np.nan

def infer_species(text):
    if any(k in text for k in ["human", "patient", "clinical", "cohort", "trial"]):
        return "human"
    if any(k in text for k in ["mouse", "mice", "murine", "xenograft"]):
        return "mouse"
    if "rat" in text:
        return "rat"
    if "dog" in text:
        return "dog"
    return ""

def infer_outcome_level(species, outcome_label, model_type):
    if species == "human" and outcome_label:
        return "human_clinical"
    if species in ["mouse", "rat", "dog"] and outcome_label:
        return "animal_in_vivo"
    if model_type:
        return "in_vitro"
    return ""

def infer_effect_direction(text):
    if any(k in text for k in ["decrease", "reduced", "reduction", "inhibition", "suppression", "cytotoxic"]):
        return "decrease"
    if any(k in text for k in ["increase", "elevated", "enhanced", "promoted"]):
        return "increase"
    return ""

def infer_evidence_weight(model_type, species, human_outcome, animal_outcome):
    if model_type and human_outcome:
        return "high"
    if model_type and animal_outcome:
        return "medium"
    if model_type:
        return "low"
    return ""

def infer_species_relevance(species):
    if species == "human":
        return 1.0
    if species in ["mouse", "rat"]:
        return 0.6
    if species == "dog":
        return 0.7
    return np.nan

def infer_translation_confidence(row):
    score = 0.0
    if row["drug_name_standard"]: score += 0.15
    if row["vitro_system_class"]: score += 0.15
    if row["tissue_context"]: score += 0.10
    if row["disease_context"]: score += 0.05
    if row["assay_type"]: score += 0.10
    if row["assay_endpoint"]: score += 0.10
    if not pd.isna(row["dose_normalized_uM"]): score += 0.10
    if not pd.isna(row["exposure_time_hours"]): score += 0.05
    if not pd.isna(row["response_value_standard"]): score += 0.10
    if row["invivo_outcome_label"] or row["human_outcome_label"]: score += 0.10
    if not pd.isna(row["replicate_count"]): score += 0.05
    return round(min(score, 0.99), 2)

def build_translation_rows(extract_df):
    rows = []

    for _, r in extract_df.iterrows():
        full_text = clean_text(r.get("pmc_fulltext", ""))
        title = clean_text(r.get("title", ""))
        abstract = clean_text(r.get("abstract", ""))

        text = f"{title} {abstract} {full_text[:50000]}"
        text_lower = text.lower()

        drug_name = find_first_pattern(text, DRUG_PATTERNS)
        cell_line = find_first_pattern(text, CELL_LINE_PATTERNS)
        tissue_context = find_rule_label(text_lower, TISSUE_RULES)
        disease_context = find_rule_label(text_lower, DISEASE_RULES)
        model_type = find_rule_label(text_lower, MODEL_RULES)
        assay_type = find_rule_label(text_lower, ASSAY_RULES)
        outcome_label = find_rule_label(text_lower, OUTCOME_RULES)
        species = infer_species(text_lower)

        ic50_uM, _ = extract_ic50(text)
        dose_uM, _ = extract_general_dose(text)
        exposure_time_hours = extract_exposure_time(text)
        viability_percent = extract_viability_percent(text)
        replicate_count = extract_replicate_count(text)
        effect_direction = infer_effect_direction(text_lower)

        if not pd.isna(ic50_uM):
            assay_endpoint = "IC50"
            response_value_standard = ic50_uM
            response_metric_standard = "IC50_uM"
            dose_normalized_uM = ic50_uM if pd.isna(dose_uM) else dose_uM
        elif not pd.isna(viability_percent):
            assay_endpoint = "viability_percent"
            response_value_standard = viability_percent
            response_metric_standard = "viability_percent"
            dose_normalized_uM = dose_uM
        else:
            assay_endpoint = ""
            response_value_standard = np.nan
            response_metric_standard = ""
            dose_normalized_uM = dose_uM

        invivo_species = species if species and species != "human" else ""
        human_outcome_label = outcome_label if species == "human" else ""
        invivo_outcome_label = outcome_label if invivo_species else ""

        outcome_level = infer_outcome_level(species, outcome_label, model_type)
        evidence_weight = infer_evidence_weight(model_type, species, human_outcome_label, invivo_outcome_label)
        species_relevance_score = infer_species_relevance(species)

        row = {
            "record_id": f"PMID_{r.get('pmid','')}",
            "study_id": r.get("paper_id", ""),
            "source_database": r.get("source_database", "PubMed/PMC"),
            "source_url": r.get("pubmed_url", ""),
            "data_added_date": r.get("data_added_date", ""),
            "curated_by": "auto_row_builder_v1",
            "review_status": "auto_extracted",
            "study_type": r.get("manual_true_study_type", ""),

            "drug_name_standard": drug_name,
            "compound_id_pubchem": "",
            "compound_id_chembl": "",
            "smiles": "",
            "intervention_type": "small_molecule" if drug_name else "",

            "vitro_system_class": model_type,
            "tissue_context": tissue_context,
            "disease_context": disease_context,
            "cell_line_name": cell_line,

            "assay_type": assay_type,
            "assay_endpoint": assay_endpoint,
            "dose_normalized_uM": dose_normalized_uM,
            "exposure_time_hours": exposure_time_hours,

            "response_value_standard": response_value_standard,
            "response_metric_standard": response_metric_standard,
            "effect_direction": effect_direction,
            "replicate_count": replicate_count,

            "invivo_species": invivo_species,
            "invivo_outcome_label": invivo_outcome_label,
            "human_outcome_label": human_outcome_label,

            "outcome_level": outcome_level,
            "species_relevance_score": species_relevance_score,
            "evidence_weight": evidence_weight,
            "translation_confidence_score": np.nan,

            "paper_id": r.get("paper_id", ""),
            "pmid": r.get("pmid", ""),
            "pmcid": r.get("pmcid", ""),
            "title": title,
            "abstract": abstract,
            "pmc_fulltext_status": r.get("pmc_fulltext_status", ""),
        }

        row["translation_confidence_score"] = infer_translation_confidence(row)
        rows.append(row)

    return pd.DataFrame(rows)
