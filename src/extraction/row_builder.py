import re
import pandas as pd
import numpy as np


DRUG_SUFFIX_RE = r"\b[A-Z][a-zA-Z0-9\-]{2,}(?:mab|nib|tinib|fenib|parib|platin|rubicin|taxel|ciclib|mycin|cycline)\b"

KNOWN_DRUGS = [
    "cisplatin", "doxorubicin", "paclitaxel", "tamoxifen", "sorafenib",
    "lenvatinib", "imatinib", "acetaminophen", "berberine", "olaparib",
    "niraparib", "rucaparib", "carboplatin", "etoposide", "gemcitabine",
    "5-fluorouracil", "fluorouracil", "5-fu", "rapamycin", "metformin",
    "docetaxel", "oxaliplatin", "irinotecan", "topotecan", "gefitinib",
    "erlotinib", "osimertinib", "afatinib", "crizotinib", "ceritinib",
    "alectinib", "vemurafenib", "dabrafenib", "trametinib", "palbociclib",
    "ribociclib", "abemaciclib", "bortezomib", "everolimus", "temsirolimus"
]

MODEL_KEYWORDS = {
    "organoid": ["organoid", "organoids"],
    "2d": ["2d", "monolayer", "cell line"],
    "3d_spheroid": ["spheroid", "3d culture", "three-dimensional"],
    "mps": ["microphysiological", "organ-on-chip", "organ on chip", "microfluidic"],
}

TISSUE_KEYWORDS = {
    "liver": ["liver", "hepatic", "hepatocyte", "hcc", "hepatocellular"],
    "lung": ["lung", "nsclc"],
    "brain": ["brain", "glioma", "glioblastoma", "medulloblastoma"],
    "kidney": ["kidney", "renal", "podocyte"],
    "breast": ["breast", "mcf-7", "mcf7"],
    "ovary": ["ovarian", "ovary"],
    "pancreas": ["pancreatic", "pancreas"],
    "stomach": ["gastric", "stomach"],
    "colon": ["colon", "colorectal", "intestinal"],
}

ASSAY_KEYWORDS = {
    "viability": ["viability", "cell viability", "survival"],
    "toxicity": ["toxicity", "cytotoxicity", "dili", "hepatotoxicity"],
    "proliferation": ["proliferation", "growth"],
    "apoptosis": ["apoptosis", "cell death"],
    "screening": ["screen", "screening", "drug response", "drug-response"],
}


def clean(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def split_sentences(text):
    text = clean(text)
    return re.split(r"(?<=[.!?])\s+", text)


def detect_label(text, vocab):
    t = text.lower()
    for label, kws in vocab.items():
        if any(k in t for k in kws):
            return label
    return ""


def detect_drugs(text):
    found = set()
    low = text.lower()

    for d in KNOWN_DRUGS:
        if d.lower() in low:
            if d.lower() == "5-fu":
                found.add("5-fluorouracil")
            else:
                found.add(d.lower())

    for m in re.finditer(DRUG_SUFFIX_RE, text):
        found.add(m.group(0).lower())

    # normalize duplicate synonym
    if "5-fluorouracil" in found and "fluorouracil" in found:
        found.discard("fluorouracil")

    return sorted(found)


def normalize_conc_to_uM(value, unit):
    try:
        value = float(value)
    except Exception:
        return np.nan

    unit = str(unit).replace("μ", "µ")
    if unit in ["µM", "uM", "UM"]:
        return value
    if unit == "nM":
        return value / 1000.0
    if unit == "mM":
        return value * 1000.0
    return np.nan


def extract_doses(text):
    out = []
    pattern = r"(\d+(?:\.\d+)?)\s*(nM|uM|µM|μM|mM)"
    for m in re.finditer(pattern, text, re.I):
        raw_val = float(m.group(1))
        raw_unit = m.group(2).replace("μ", "µ")
        out.append({
            "dose_raw": f"{raw_val} {raw_unit}",
            "dose_normalized_uM": normalize_conc_to_uM(raw_val, raw_unit)
        })
    return out


def extract_time_hours(text):
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours)\b",
        r"(\d+(?:\.\d+)?)\s*(day|days)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()
            if unit in ["day", "days"]:
                return val * 24.0
            return val
    return np.nan


def extract_response(text):
    # IC50 / EC50 / GI50 = 1.2 uM
    m = re.search(
        r"\b(IC50|IC\s*50|EC50|EC\s*50|GI50|GI\s*50)\b\s*(?:was|were|=|:|of|approximately|about)?\s*([<>~]?\s*\d+(?:\.\d+)?)\s*(nM|uM|µM|μM|mM)",
        text,
        re.I
    )
    if m:
        metric = re.sub(r"\s+", "", m.group(1).upper())
        value_raw = m.group(2).replace(" ", "").replace("<", "").replace(">", "").replace("~", "")
        val = normalize_conc_to_uM(float(value_raw), m.group(3).replace("μ", "µ"))
        return metric, val, f"{metric}_uM"

    # IC50 values ranged from 0.1 to 5 uM
    m = re.search(
        r"\b(IC50|IC\s*50|EC50|EC\s*50|GI50|GI\s*50)\b.*?(\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(nM|uM|µM|μM|mM)",
        text,
        re.I
    )
    if m:
        metric = re.sub(r"\s+", "", m.group(1).upper())
        avg_raw = (float(m.group(2)) + float(m.group(3))) / 2.0
        val = normalize_conc_to_uM(avg_raw, m.group(4).replace("μ", "µ"))
        return metric, val, f"{metric}_uM"

    # 50% viability / 70% inhibition / 40% reduction
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s*(viability|cell viability|inhibition|growth inhibition|reduction|survival|cell death|apoptosis)",
        text,
        re.I
    )
    if m:
        metric = m.group(2).lower().replace(" ", "_")
        return metric, float(m.group(1)), "percent"

    # viability reduced by 40%
    m = re.search(
        r"(viability|cell viability|growth|proliferation|survival|apoptosis|cell death).*?(reduced|decreased|increased|inhibited|suppressed|enhanced).*?(\d+(?:\.\d+)?)\s*%",
        text,
        re.I
    )
    if m:
        metric = m.group(1).lower().replace(" ", "_")
        return metric, float(m.group(3)), "percent"

    # reduced viability to 35%
    m = re.search(
        r"(reduced|decreased|increased|inhibited|suppressed|enhanced).*?(viability|cell viability|growth|proliferation|survival|apoptosis|cell death).*?(\d+(?:\.\d+)?)\s*%",
        text,
        re.I
    )
    if m:
        metric = m.group(2).lower().replace(" ", "_")
        return metric, float(m.group(3)), "percent"

    return "", np.nan, ""


def infer_effect_direction(text):
    low = text.lower()
    if any(k in low for k in ["decreased", "reduced", "reduction", "inhibited", "suppressed", "cytotoxic", "killed", "lowered"]):
        return "decrease"
    if any(k in low for k in ["increased", "enhanced", "promoted", "elevated"]):
        return "increase"
    return ""


def infer_outcome_level(text, model_type):
    low = text.lower()
    if any(k in low for k in ["patient", "clinical", "human cohort", "trial"]):
        return "human_clinical"
    if any(k in low for k in ["mouse", "mice", "murine", "rat", "xenograft", "in vivo"]):
        return "animal_in_vivo"
    if model_type:
        return "in_vitro"
    return ""


def infer_species(text):
    low = text.lower()
    if any(k in low for k in ["mouse", "mice", "murine", "xenograft"]):
        return "mouse"
    if "rat" in low:
        return "rat"
    if any(k in low for k in ["patient", "clinical", "human"]):
        return "human"
    return ""


def confidence(row):
    score = 0.0
    weights = {
        "drug_name_standard": 0.20,
        "vitro_system_class": 0.15,
        "tissue_context": 0.10,
        "assay_type": 0.10,
        "dose_normalized_uM": 0.15,
        "exposure_time_hours": 0.05,
        "response_value_standard": 0.20,
        "outcome_level": 0.05,
    }

    for col, weight in weights.items():
        val = row.get(col, "")
        if not pd.isna(val) and str(val).strip() != "":
            score += weight

    return round(min(score, 0.99), 2)


def build_translation_rows(extract_df):
    rows = []

    for _, paper in extract_df.iterrows():
        paper_id = paper.get("paper_id", "")
        pmid = paper.get("pmid", "")
        title = clean(paper.get("title", ""))
        abstract = clean(paper.get("abstract", ""))

        priority_text = clean(paper.get("pmc_priority_text", ""))
        fulltext = clean(paper.get("pmc_fulltext", ""))

        if priority_text:
            text = f"{title}. {abstract}. {priority_text[:120000]}"
        else:
            text = f"{title}. {abstract}. {fulltext[:120000]}"

        if not text.strip():
            text = f"{title}. {abstract}."

        sentences = split_sentences(text)

        for i, sent in enumerate(sentences):
            window = " ".join(sentences[max(0, i - 2): min(len(sentences), i + 3)])

            drugs = detect_drugs(window)
            doses = extract_doses(window)
            resp_metric, resp_value, resp_standard = extract_response(window)

            if not drugs:
                continue

            # Require at least dose OR response.
            if not doses and pd.isna(resp_value):
                continue

            model = detect_label(window, MODEL_KEYWORDS)
            tissue = detect_label(window, TISSUE_KEYWORDS)
            assay = detect_label(window, ASSAY_KEYWORDS)
            exposure = extract_time_hours(window)
            effect = infer_effect_direction(window)
            species = infer_species(window)
            outcome_level = infer_outcome_level(window, model)

            if not doses:
                doses = [{"dose_raw": "", "dose_normalized_uM": np.nan}]

            for drug in drugs:
                for d in doses:
                    row = {
                        "record_id": f"PMID_{pmid}_{len(rows) + 1:06d}",
                        "study_id": paper_id,
                        "source_database": paper.get("source_database", "PubMed/PMC"),
                        "source_url": paper.get("pubmed_url", ""),
                        "data_added_date": paper.get("data_added_date", ""),
                        "curated_by": "auto_multirow_fulltext_extractor_v2",
                        "review_status": "auto_extracted",
                        "study_type": paper.get("manual_true_study_type", ""),

                        "drug_name_standard": drug,
                        "compound_id_pubchem": "",
                        "compound_id_chembl": "",
                        "smiles": "",
                        "intervention_type": "small_molecule",

                        "vitro_system_class": model,
                        "tissue_context": tissue,
                        "disease_context": "",
                        "cell_line_name": "",

                        "assay_type": assay,
                        "assay_endpoint": resp_metric,
                        "dose_normalized_uM": d["dose_normalized_uM"],
                        "exposure_time_hours": exposure,

                        "response_value_standard": resp_value,
                        "response_metric_standard": resp_standard,
                        "effect_direction": effect,
                        "replicate_count": np.nan,

                        "invivo_species": species if species != "human" else "",
                        "invivo_outcome_label": "",
                        "human_outcome_label": "clinical_context" if species == "human" else "",

                        "outcome_level": outcome_level,
                        "species_relevance_score": 1.0 if species == "human" else (0.6 if species in ["mouse", "rat"] else np.nan),
                        "evidence_weight": "high" if species == "human" else ("medium" if species in ["mouse", "rat"] else "low"),
                        "translation_confidence_score": np.nan,

                        "paper_id": paper_id,
                        "pmid": pmid,
                        "pmcid": paper.get("pmcid", ""),
                        "title": title,
                        "abstract": abstract,
                        "pmc_fulltext_status": paper.get("pmc_fulltext_status", ""),
                        "evidence_sentence": sent,
                        "evidence_window": window,
                    }

                    row["translation_confidence_score"] = confidence(row)
                    rows.append(row)

    return pd.DataFrame(rows)
