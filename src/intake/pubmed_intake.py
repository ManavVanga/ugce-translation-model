import os
import re
import time
import json
import requests
import pandas as pd
from datetime import datetime
from xml.etree import ElementTree as ET

from src.utils.config import get_base_output_dir


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

RETMAX = 20
SLEEP = 0.34


SEARCH_QUERY = (
    '("organoid" OR "spheroid" OR "3D culture" OR "cell line" OR '
    '"microphysiological system" OR "organ-on-chip" OR "xenograft" OR "PDX" OR "in vivo") '
    'AND '
    '("drug response" OR "dose response" OR "IC50" OR "EC50" OR "GI50" OR '
    '"viability" OR "inhibition" OR "tumor growth" OR "toxicity" OR "clinical response")'
)


def clean_text(x):
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def yesno(value):
    return "yes" if value else "no"


def contains_any(text, keywords):
    text = text.lower()
    return any(k.lower() in text for k in keywords)


def extract_endpoint_tags(text):
    tags = []

    endpoint_map = {
        "drug_response": ["drug response", "dose response", "dose-response", "ic50", "ec50", "gi50"],
        "viability": ["viability", "cell viability", "survival"],
        "toxicity_safety": ["toxicity", "cytotoxicity", "hepatotoxicity", "dili", "adverse"],
        "tumor_growth": ["tumor growth", "tumor volume", "tumor inhibition", "tumor regression"],
        "apoptosis_cell_death": ["apoptosis", "cell death"],
        "proliferation_growth": ["proliferation", "growth"],
        "migration_invasion": ["migration", "invasion", "metastasis"],
        "immune_response": ["immune", "t cell", "macrophage", "cytokine"],
        "transcriptomics": ["rna-seq", "rnaseq", "transcriptomic", "transcriptomics"],
        "proteomics": ["proteomic", "proteomics", "protein expression"],
        "metabolomics": ["metabolomic", "metabolomics"],
        "pk_pd": ["pharmacokinetic", "pharmacodynamic", "auc", "cmax", "half-life"],
        "clinical_efficacy": ["overall response rate", "orr", "progression-free survival", "pfs", "overall survival", "os"],
        "clinical_safety": ["adverse event", "toxicity", "dose-limiting toxicity"],
    }

    for tag, keys in endpoint_map.items():
        if contains_any(text, keys):
            tags.append(tag)

    return sorted(set(tags))


def compute_universal_flags(title, abstract):
    text = f"{clean_text(title)} {clean_text(abstract)}".lower()

    has_drug_exposure = contains_any(text, [
        "drug", "compound", "inhibitor", "treatment", "therapy", "therapeutic",
        "chemotherapy", "small molecule", "antibody", "car-t", "car t"
    ])

    has_dose_signal = bool(re.search(
        r"(\d+(\.\d+)?\s*(nm|µm|μm|um|mm|mg/kg|mg kg|ng/ml|µg/ml|ug/ml))",
        text,
        re.I
    )) or contains_any(text, [
        "dose", "dosage", "concentration", "dose-response", "dose response",
        "concentration-dependent", "concentration dependent"
    ])

    has_response_signal = contains_any(text, [
        "ic50", "ec50", "gi50", "viability", "inhibition", "inhibited",
        "reduction", "reduced", "decrease", "decreased", "increase", "increased",
        "response", "resistance", "sensitive", "sensitivity", "cytotoxicity",
        "tumor growth inhibition", "tumor regression", "apoptosis"
    ]) or bool(re.search(r"\d+(\.\d+)?\s*%", text))

    has_model_invitro = contains_any(text, [
        "in vitro", "cell line", "2d", "monolayer", "3d culture", "spheroid",
        "organoid", "organoids", "microphysiological", "organ-on-chip",
        "organ on chip", "primary cells", "co-culture", "coculture"
    ])

    has_model_animal = contains_any(text, [
        "in vivo", "mouse", "mice", "murine", "rat", "xenograft",
        "pdx", "patient-derived xenograft", "animal model"
    ])

    has_model_human = contains_any(text, [
        "patient", "patients", "clinical", "clinical trial", "cohort",
        "human", "phase i", "phase ii", "phase iii"
    ])

    has_omics_signal = contains_any(text, [
        "rna-seq", "rnaseq", "transcriptomic", "transcriptomics",
        "proteomic", "proteomics", "metabolomic", "metabolomics",
        "single-cell", "scrna", "gene expression", "pathway"
    ])

    has_toxicity_signal = contains_any(text, [
        "toxicity", "cytotoxicity", "hepatotoxicity", "nephrotoxicity",
        "cardiotoxicity", "dili", "adverse", "safety"
    ])

    has_efficacy_signal = contains_any(text, [
        "efficacy", "response", "tumor regression", "tumor growth inhibition",
        "sensitive", "sensitivity", "resistance", "survival"
    ])

    has_pkpd_signal = contains_any(text, [
        "pharmacokinetic", "pharmacodynamics", "pharmacodynamic",
        "auc", "cmax", "half-life", "clearance", "bioavailability"
    ])

    has_clinical_endpoint = contains_any(text, [
        "overall survival", "progression-free survival", "orr",
        "objective response", "clinical response", "adverse event",
        "dose-limiting toxicity", "phase i", "phase ii", "phase iii"
    ])

    score = 0
    score += 2 if has_drug_exposure else 0
    score += 2 if has_dose_signal else 0
    score += 3 if has_response_signal else 0
    score += 2 if has_model_invitro else 0
    score += 2 if has_model_animal else 0
    score += 2 if has_model_human else 0
    score += 1 if has_omics_signal else 0
    score += 1 if has_toxicity_signal else 0
    score += 1 if has_efficacy_signal else 0
    score += 1 if has_pkpd_signal else 0
    score += 3 if has_clinical_endpoint else 0

    has_any_model = has_model_invitro or has_model_animal or has_model_human

    high_priority = (
        has_drug_exposure
        and has_dose_signal
        and has_response_signal
        and has_any_model
    )

    if high_priority:
        bucket = "HIGH_PRIORITY"
    elif score >= 7:
        bucket = "MEDIUM_PRIORITY"
    else:
        bucket = "LOW_PRIORITY"

    endpoint_tags = extract_endpoint_tags(text)

    return {
        "has_drug_exposure": yesno(has_drug_exposure),
        "has_dose_signal": yesno(has_dose_signal),
        "has_response_signal": yesno(has_response_signal),
        "has_model_invitro": yesno(has_model_invitro),
        "has_model_animal": yesno(has_model_animal),
        "has_model_human": yesno(has_model_human),
        "has_omics_signal": yesno(has_omics_signal),
        "has_toxicity_signal": yesno(has_toxicity_signal),
        "has_efficacy_signal": yesno(has_efficacy_signal),
        "has_pkpd_signal": yesno(has_pkpd_signal),
        "has_clinical_endpoint": yesno(has_clinical_endpoint),
        "universal_priority_score": score,
        "universal_priority_bucket": bucket,
        "endpoint_signal_tags": "|".join(endpoint_tags),
        "manual_keep_decision": "keep_high_priority" if high_priority else "auto_not_ready",
        "manual_extract_ready": "yes" if high_priority else "no",
        "manual_reason": (
            "auto-selected: drug+dose+response+model signal"
            if high_priority
            else "auto-screened but not high-priority"
        ),
    }


def infer_study_type(flags):
    invitro = flags["has_model_invitro"] == "yes"
    animal = flags["has_model_animal"] == "yes"
    human = flags["has_model_human"] == "yes"

    if invitro and animal and human:
        return "invitro_animal_human"
    if invitro and animal:
        return "invitro_animal"
    if invitro and human:
        return "invitro_human"
    if animal and human:
        return "animal_human"
    if invitro:
        return "invitro_only"
    if animal:
        return "animal_only"
    if human:
        return "human_only"
    return "unclear"


def pubmed_search(query, retmax=RETMAX):
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "pub date",
    }
    r = requests.get(ESEARCH_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])


def pubmed_fetch(pmids):
    if not pmids:
        return ""

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    r = requests.get(EFETCH_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.text


def parse_pubmed_xml(xml_text):
    root = ET.fromstring(xml_text)
    rows = []

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        article_node = medline.find("Article") if medline is not None else None

        pmid = clean_text(medline.findtext("PMID")) if medline is not None else ""
        title = clean_text(article_node.findtext("ArticleTitle")) if article_node is not None else ""

        abstract_parts = []
        if article_node is not None:
            for ab in article_node.findall(".//AbstractText"):
                label = ab.attrib.get("Label", "")
                text = clean_text("".join(ab.itertext()))
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = clean_text(" ".join(abstract_parts))

        journal = ""
        year = ""
        if article_node is not None:
            journal = clean_text(article_node.findtext(".//Journal/Title"))
            year = clean_text(article_node.findtext(".//PubDate/Year"))

        doi = ""
        for aid in article.findall(".//ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                doi = clean_text(aid.text)

        authors = []
        if article_node is not None:
            for au in article_node.findall(".//Author"):
                last = clean_text(au.findtext("LastName"))
                fore = clean_text(au.findtext("ForeName"))
                name = clean_text(f"{fore} {last}")
                if name:
                    authors.append(name)

        flags = compute_universal_flags(title, abstract)
        study_type = infer_study_type(flags)

        rows.append({
            "paper_id": f"PMID_{pmid}",
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "journal": journal,
            "year": year,
            "authors": "; ".join(authors[:20]),
            "abstract": abstract,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "search_query": SEARCH_QUERY,
            "candidate_class": "universal_iviv_candidate",
            "manual_screen_status": "auto_screened",
            "manual_priority": flags["universal_priority_bucket"],
            "manual_notes": "",
            "data_added_date": datetime.now().strftime("%Y-%m-%d"),
            "curated_by": "auto_universal_endpoint_screening_v1",
            "manual_true_study_type": study_type,
            **flags,
        })

    return pd.DataFrame(rows)


def merge_with_existing(new_df, registry_file, screening_file):
    if os.path.exists(registry_file):
        old_df = pd.read_csv(registry_file)
    else:
        old_df = pd.DataFrame()

    if len(old_df) > 0:
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["paper_id"], keep="first")
    else:
        combined = new_df.copy()

    combined.to_csv(registry_file, index=False)
    combined.to_csv(screening_file, index=False)

    return combined


def write_audit_sample(df, out_dir, n=20):
    audit_dir = os.path.join(out_dir, "audit")
    os.makedirs(audit_dir, exist_ok=True)

    if len(df) == 0:
        return

    sample_n = min(n, len(df))
    sample_df = df.sample(n=sample_n, random_state=42)
    sample_path = os.path.join(audit_dir, "audit_sample_v1.csv")
    sample_df.to_csv(sample_path, index=False)
    print("Saved audit sample:", sample_path)


def write_log(base_dir, num_records, high_priority_count):
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, "data_collection_log.csv")

    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame(columns=[
            "log_id", "date_collected", "dataset_name", "source_type",
            "study_type", "num_records_added", "curated_by", "notes"
        ])

    new_log = pd.DataFrame([{
        "log_id": f"LOG_{len(log_df) + 1:03d}",
        "date_collected": datetime.now().strftime("%Y-%m-%d"),
        "dataset_name": "pubmed_paper_registry_v1",
        "source_type": "pubmed_universal_endpoint_intake",
        "study_type": "universal_iviv_screening",
        "num_records_added": int(num_records),
        "curated_by": "auto_universal_endpoint_screening_v1",
        "notes": f"High priority auto-extract-ready papers={high_priority_count}"
    }])

    log_df = pd.concat([log_df, new_log], ignore_index=True)
    log_df.to_csv(log_file, index=False)


if __name__ == "__main__":
    base_dir = get_base_output_dir()

    registry_dir = os.path.join(base_dir, "collection", "literature_registry")
    os.makedirs(registry_dir, exist_ok=True)

    registry_file = os.path.join(registry_dir, "pubmed_paper_registry_v1.csv")
    screening_file = os.path.join(registry_dir, "pubmed_paper_screening_v1.csv")

    print("Loading config...")
    print("Base output dir:", base_dir)

    print("\nSearching PubMed...")
    pmids = pubmed_search(SEARCH_QUERY, RETMAX)
    print("PMIDs fetched:", len(pmids))
    print(pmids)

    time.sleep(SLEEP)

    print("\nFetching PubMed details...")
    xml_text = pubmed_fetch(pmids)

    print("\nParsing PubMed XML...")
    new_df = parse_pubmed_xml(xml_text)

    high_priority_count = (
        new_df["universal_priority_bucket"].astype(str).eq("HIGH_PRIORITY").sum()
        if len(new_df) > 0 else 0
    )

    print("New records parsed:", len(new_df))
    print("High priority records:", high_priority_count)

    combined_df = merge_with_existing(new_df, registry_file, screening_file)

    write_audit_sample(combined_df, registry_dir, n=20)
    write_log(base_dir, len(new_df), high_priority_count)

    print("\nSaved files:")
    print(registry_file)
    print(screening_file)

    show_cols = [
        "paper_id", "year", "title",
        "has_drug_exposure", "has_dose_signal", "has_response_signal",
        "has_model_invitro", "has_model_animal", "has_model_human",
        "universal_priority_score", "universal_priority_bucket",
        "manual_extract_ready", "endpoint_signal_tags"
    ]
    show_cols = [c for c in show_cols if c in combined_df.columns]

    print("\nPreview:")
    print(combined_df[show_cols].head(20).to_string(index=False))
