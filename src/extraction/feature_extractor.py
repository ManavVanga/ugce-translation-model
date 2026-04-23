import pandas as pd
import re


def extract_ic50(text):
    match = re.search(r'IC50\s*[:=]?\s*(\d+\.?\d*)\s*(uM|µM|nM|mM)?', text, re.IGNORECASE)
    if match:
        return float(match.group(1)), match.group(2)
    return None, None


def extract_viability(text):
    match = re.search(r'(\d+\.?\d*)\s*%\s*(viability|cell viability)', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def detect_model_type(text):
    text = text.lower()

    if "organoid" in text:
        return "organoid"
    if "organ-on-chip" in text or "organ on chip" in text:
        return "organ_on_chip"
    if "microphysiological" in text:
        return "mps"
    if "2d culture" in text or "cell line" in text:
        return "2d"
    if "spheroid" in text:
        return "3d_spheroid"

    return "unknown"


def detect_species(text):
    text = text.lower()

    if "mouse" in text or "mice" in text:
        return "mouse"
    if "rat" in text:
        return "rat"
    if "human" in text or "patient" in text:
        return "human"

    return "unknown"


def build_features(df):
    feature_rows = []

    for _, row in df.iterrows():
        text = f"{row.get('title','')} {row.get('abstract','')}"

        ic50, ic50_unit = extract_ic50(text)
        viability = extract_viability(text)
        model_type = detect_model_type(text)
        species = detect_species(text)

        feature_rows.append({
            "paper_id": row["paper_id"],
            "pmid": row["pmid"],
            "drug_text": text[:200],
            "model_type": model_type,
            "species": species,
            "ic50": ic50,
            "ic50_unit": ic50_unit,
            "viability_percent": viability
        })

    return pd.DataFrame(feature_rows)
