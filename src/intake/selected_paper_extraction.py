import os
import re
import json
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from src.utils.config import get_base_output_dir

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
BIOC_PMC_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
SLEEP_BETWEEN_CALLS = 0.3


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()

def fetch_pubmed_xml_by_pmid(pmid):
    url = f"{EUTILS_BASE}/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": str(pmid),
        "retmode": "xml"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.text

def extract_abstract_text(article_node):
    parts = []
    for abst in article_node.findall(".//Abstract/AbstractText"):
        txt = "".join(abst.itertext()).strip()
        if txt:
            parts.append(txt)
    return " ".join(parts).strip()

def parse_pubmed_article(xml_text):
    root = ET.fromstring(xml_text)
    article = root.find(".//PubmedArticle")
    if article is None:
        return {}

    pmid = clean_text(article.findtext(".//PMID"))
    title = clean_text(article.findtext(".//ArticleTitle"))
    journal = clean_text(article.findtext(".//Journal/Title"))
    year = clean_text(article.findtext(".//PubDate/Year"))

    if not year:
        medline_date = clean_text(article.findtext(".//PubDate/MedlineDate"))
        year = medline_date[:4] if medline_date else ""

    abstract = clean_text(extract_abstract_text(article))

    doi = ""
    for aid in article.findall(".//ArticleId"):
        if aid.attrib.get("IdType") == "doi":
            doi = clean_text(aid.text)
            break

    authors = []
    for a in article.findall(".//Author"):
        ln = a.findtext("LastName")
        ini = a.findtext("Initials")
        if ln:
            authors.append(f"{ln} {ini or ''}".strip())

    return {
        "pmid": pmid,
        "title": title,
        "journal": journal,
        "year": year,
        "doi": doi,
        "authors": "; ".join(authors[:15]),
        "abstract": abstract,
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    }

def fetch_pmcid_from_pmid(pmid):
    params = {
        "ids": str(pmid),
        "idtype": "pmid",
        "format": "json"
    }
    r = requests.get(PMC_IDCONV_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    records = data.get("records", [])
    if records and isinstance(records, list):
        rec = records[0]
        return clean_text(rec.get("pmcid", ""))
    return ""

def fetch_pmc_bioc_fulltext(pmcid):
    if not pmcid:
        return "", "no_pmcid"

    pmcid_num = pmcid.replace("PMC", "")
    url = BIOC_PMC_URL.format(pmcid=pmcid_num)

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return "", f"http_{r.status_code}"

        data = r.json()
        passages = []

        for doc in data.get("documents", []):
            for p in doc.get("passages", []):
                txt = clean_text(p.get("text", ""))
                if txt:
                    passages.append(txt)

        fulltext = "\n".join(passages).strip()
        if fulltext:
            return fulltext, "ok"
        return "", "empty"
    except Exception as e:
        return "", f"error_{type(e).__name__}"


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    base_dir = get_base_output_dir()

    registry_dir = os.path.join(base_dir, "collection", "literature_registry")
    extraction_dir = os.path.join(base_dir, "collection", "literature_extraction")
    logs_dir = os.path.join(base_dir, "logs")

    os.makedirs(extraction_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    screen_file = os.path.join(registry_dir, "pubmed_paper_screening_v1.csv")
    extraction_file = os.path.join(extraction_dir, "selected_paper_extraction_v1.csv")
    log_file = os.path.join(logs_dir, "data_collection_log.csv")

    if not os.path.exists(screen_file):
        raise FileNotFoundError(f"Screening file not found: {screen_file}")

    screen_df = pd.read_csv(screen_file)

    keep_df = screen_df[
        (screen_df["manual_keep_decision"].astype(str).str.lower() == "keep_high_priority") &
        (screen_df["manual_extract_ready"].astype(str).str.lower() == "yes")
    ].copy()

    print("Selected papers for extraction:", len(keep_df))

    if len(keep_df) == 0:
        print("No papers marked for extraction. Exiting.")
        raise SystemExit(0)

    rows = []

    for i, (_, row) in enumerate(keep_df.iterrows(), start=1):
        pmid = clean_text(row.get("pmid", ""))
        print(f"Extracting {i}/{len(keep_df)} | PMID={pmid}")

        meta = {
            "pmid": pmid,
            "title": clean_text(row.get("title", "")),
            "journal": clean_text(row.get("journal", "")),
            "year": clean_text(row.get("year", "")),
            "doi": clean_text(row.get("doi", "")),
            "authors": clean_text(row.get("authors", "")),
            "abstract": clean_text(row.get("abstract", "")),
            "pubmed_url": clean_text(row.get("pubmed_url", "")),
        }

        if not meta["abstract"] or not meta["title"]:
            try:
                pubmed_xml = fetch_pubmed_xml_by_pmid(pmid)
                meta = parse_pubmed_article(pubmed_xml)
            except Exception:
                pass

        time.sleep(SLEEP_BETWEEN_CALLS)

        pmcid = ""
        fulltext = ""
        fulltext_status = "not_attempted"

        try:
            pmcid = fetch_pmcid_from_pmid(pmid)
            time.sleep(SLEEP_BETWEEN_CALLS)

            if pmcid:
                fulltext, fulltext_status = fetch_pmc_bioc_fulltext(pmcid)
                time.sleep(SLEEP_BETWEEN_CALLS)
            else:
                fulltext_status = "no_pmcid"
        except Exception as e:
            fulltext_status = f"error_{type(e).__name__}"

        rows.append({
            "paper_id": row.get("paper_id", ""),
            "pmid": pmid,
            "pmcid": pmcid,
            "doi": meta.get("doi", ""),
            "title": meta.get("title", ""),
            "journal": meta.get("journal", ""),
            "year": meta.get("year", ""),
            "authors": meta.get("authors", ""),
            "pubmed_url": meta.get("pubmed_url", ""),
            "source_database": "PubMed/PMC",
            "manual_true_study_type": row.get("manual_true_study_type", ""),
            "manual_keep_decision": row.get("manual_keep_decision", ""),
            "manual_reason": row.get("manual_reason", ""),
            "abstract": meta.get("abstract", ""),
            "pmc_fulltext_status": fulltext_status,
            "pmc_fulltext": fulltext,
            "manual_extract_ready": row.get("manual_extract_ready", ""),
            "data_added_date": datetime.now().strftime("%Y-%m-%d"),
            "curated_by": "Manav"
        })

    extract_df = pd.DataFrame(rows)
    extract_df.to_csv(extraction_file, index=False)

    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame(columns=[
            "log_id", "date_collected", "dataset_name", "source_type",
            "study_type", "num_records_added", "curated_by", "notes"
        ])

    new_log = pd.DataFrame([{
        "log_id": f"LOG_{len(log_df)+1:03d}",
        "date_collected": datetime.now().strftime("%Y-%m-%d"),
        "dataset_name": "selected_paper_extraction_v1",
        "source_type": "pubmed_pmc_extraction",
        "study_type": "literature_extraction",
        "num_records_added": int(len(extract_df)),
        "curated_by": "Manav",
        "notes": "Extracted selected papers into structured extraction file"
    }])

    log_df = pd.concat([log_df, new_log], ignore_index=True)
    log_df.to_csv(log_file, index=False)

    print("Saved extraction file:", extraction_file)
    print(extract_df[["paper_id", "pmid", "pmcid", "pmc_fulltext_status"]].head())
