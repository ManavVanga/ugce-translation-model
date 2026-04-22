import requests
import json
import time
import os
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from src.utils.config import load_drive_paths

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SEARCH_QUERY = '(organoid OR organ-on-chip OR microphysiological OR 2D culture OR CRISPR) AND (drug response OR IC50 OR viability OR toxicity)'
MAX_RESULTS = 20


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def search_pubmed(query, max_results=10):
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "date"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    return data["esearchresult"]["idlist"]


def fetch_details(pmids):
    url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.text


def parse_pubmed_xml(xml_text, search_query):
    root = ET.fromstring(xml_text)
    rows = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = article.findtext(".//ArticleTitle", default="")
        journal = article.findtext(".//Journal/Title", default="")
        year = article.findtext(".//PubDate/Year", default="")

        if not year:
            medline_date = article.findtext(".//PubDate/MedlineDate", default="")
            year = medline_date[:4] if medline_date else ""

        abstract_parts = []
        for abst in article.findall(".//Abstract/AbstractText"):
            text = "".join(abst.itertext()).strip()
            if text:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        doi = ""
        for aid in article.findall(".//ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                doi = (aid.text or "").strip()
                break

        rows.append({
            "paper_id": f"PMID_{pmid}",
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "journal": journal,
            "year": year,
            "abstract": abstract,
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "search_query": search_query,
            "candidate_class": "",
            "manual_screen_status": "unreviewed",
            "manual_priority": "",
            "manual_notes": "",
            "data_added_date": datetime.now().strftime("%Y-%m-%d"),
            "curated_by": "Manav"
        })

    return pd.DataFrame(rows)


def classify_candidate(title, abstract):
    text = f"{title} {abstract}".lower()

    has_invitro = any(k in text for k in [
        "organoid", "microphysiological", "organ-on-chip", "organ on chip",
        "2d culture", "cell line", "in vitro", "crispr", "spheroid"
    ])
    has_invivo = any(k in text for k in [
        "mouse", "mice", "rat", "animal model", "xenograft", "in vivo", "murine"
    ])
    has_human = any(k in text for k in [
        "patient", "clinical", "human", "trial", "cohort"
    ])

    if has_invitro and (has_invivo or has_human):
        return "combined_candidate"
    if has_invitro:
        return "in_vitro_candidate"
    if has_invivo or has_human:
        return "in_vivo_candidate"
    return "unclear_candidate"


def score_priority(title, abstract):
    text = f"{title} {abstract}".lower()
    score = 0

    if "in vivo" in text:
        score += 3
    if "patient" in text or "clinical" in text or "human" in text:
        score += 3
    if "organoid" in text:
        score += 2
    if "organ-on-chip" in text or "organ on chip" in text or "microphysiological" in text:
        score += 2
    if "drug response" in text or "toxicity" in text or "efficacy" in text:
        score += 2
    if "screen" in text or "screening" in text:
        score += 1

    if "review" in text or "models of" in text or "advancements" in text:
        score -= 2

    return score


def priority_bucket(score):
    if score >= 7:
        return "high"
    elif score >= 4:
        return "medium"
    else:
        return "low"


def infer_true_study_type(title, abstract):
    text = f"{title} {abstract}".lower()

    has_invitro = any(k in text for k in [
        "organoid", "organ-on-chip", "organ on chip",
        "microphysiological", "2d culture", "in vitro", "crispr"
    ])
    has_invivo = any(k in text for k in [
        "in vivo", "mouse", "mice", "murine", "rat", "xenograft"
    ])
    has_human = any(k in text for k in [
        "patient", "clinical", "human", "trial", "cohort"
    ])

    if has_invitro and (has_invivo or has_human):
        return "combined"
    if has_invitro:
        return "in_vitro_only"
    if has_invivo or has_human:
        return "in_vivo_only"
    return "unclear"


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Loading config...")
    paths = load_drive_paths()

    registry_dir = paths["registry_dir"]
    logs_dir = paths["logs_dir"]
    os.makedirs(registry_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    registry_file = os.path.join(registry_dir, "pubmed_paper_registry_v1.csv")
    screening_file = os.path.join(registry_dir, "pubmed_paper_screening_v1.csv")
    log_file = os.path.join(logs_dir, "data_collection_log.csv")

    if os.path.exists(registry_file):
        registry_df = pd.read_csv(registry_file)
    else:
        registry_df = pd.DataFrame()

    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame(columns=[
            "log_id", "date_collected", "dataset_name", "source_type",
            "study_type", "num_records_added", "curated_by", "notes"
        ])

    print("\nSearching PubMed...")
    pmids = search_pubmed(SEARCH_QUERY, MAX_RESULTS)
    print(f"PMIDs fetched: {len(pmids)}")

    existing_pmids = set(registry_df["pmid"].astype(str)) if len(registry_df) > 0 else set()
    new_pmids = [p for p in pmids if str(p) not in existing_pmids]

    print(f"New PMIDs: {len(new_pmids)}")

    if len(new_pmids) == 0:
        print("No new papers found. Registry unchanged.")
    else:
        print("\nFetching new paper details...")
        xml_data = fetch_details(new_pmids)
        new_df = parse_pubmed_xml(xml_data, SEARCH_QUERY)

        new_df["candidate_class"] = new_df.apply(
            lambda r: classify_candidate(str(r["title"]), str(r["abstract"])),
            axis=1
        )
        new_df["priority_score"] = new_df.apply(
            lambda r: score_priority(str(r["title"]), str(r["abstract"])),
            axis=1
        )
        new_df["priority_bucket"] = new_df["priority_score"].apply(priority_bucket)
        new_df["manual_true_study_type"] = new_df.apply(
            lambda r: infer_true_study_type(str(r["title"]), str(r["abstract"])),
            axis=1
        )
        new_df["manual_keep_decision"] = "unreviewed"
        new_df["manual_reason"] = ""
        new_df["manual_has_invitro_data"] = ""
        new_df["manual_has_animal_data"] = ""
        new_df["manual_has_human_data"] = ""
        new_df["manual_has_dose_response"] = ""
        new_df["manual_has_ic50"] = ""
        new_df["manual_has_viability"] = ""
        new_df["manual_has_translation_link"] = ""
        new_df["manual_extract_ready"] = "no"

        # auto-mark top candidates
        candidate_df = new_df[
            new_df["priority_bucket"].isin(["high", "medium"])
        ].copy()

        candidate_df = candidate_df.sort_values(
            ["priority_score", "year"], ascending=[False, False]
        ).head(10)

        selected_pmids = set(candidate_df["pmid"].astype(str))

        mask = new_df["pmid"].astype(str).isin(selected_pmids)
        new_df.loc[mask, "manual_keep_decision"] = "keep_high_priority"
        new_df.loc[mask, "manual_extract_ready"] = "yes"
        new_df.loc[mask, "manual_reason"] = "auto-marked by github intake pipeline"

        # update registry + screening
        registry_df = pd.concat([registry_df, new_df], ignore_index=True)
        registry_df = registry_df.drop_duplicates(subset=["pmid"]).reset_index(drop=True)

        registry_df.to_csv(registry_file, index=False)
        registry_df.to_csv(screening_file, index=False)

        # update log
        new_log = pd.DataFrame([{
            "log_id": f"LOG_{len(log_df)+1:03d}",
            "date_collected": datetime.now().strftime("%Y-%m-%d"),
            "dataset_name": "pubmed_paper_registry_v1",
            "source_type": "github_pubmed_intake",
            "study_type": "literature_registry_and_screening",
            "num_records_added": len(new_df),
            "curated_by": "Manav",
            "notes": f"New papers added={len(new_df)}"
        }])

        log_df = pd.concat([log_df, new_log], ignore_index=True)
        log_df.to_csv(log_file, index=False)

        print("\nSaved files:")
        print(registry_file)
        print(screening_file)

        print("\nPreview:")
        print(new_df[[
            "paper_id", "year", "title", "candidate_class",
            "priority_score", "priority_bucket", "manual_keep_decision",
            "manual_extract_ready"
        ]].head())
