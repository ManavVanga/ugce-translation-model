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
SEARCH_QUERY = "(organoid OR organ-on-chip OR microphysiological OR 2D culture OR CRISPR) AND (drug response OR IC50 OR viability OR toxicity)"
MAX_RESULTS = 5  # small for testing


# ------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------
def search_pubmed(query, max_results=10):
    url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
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


def parse_pubmed_xml(xml_text):
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
            "search_query": SEARCH_QUERY,
            "data_added_date": datetime.now().strftime("%Y-%m-%d")
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Loading config...")
    paths = load_drive_paths()

    registry_dir = paths["registry_dir"]
    os.makedirs(registry_dir, exist_ok=True)

    out_file = os.path.join(registry_dir, "pubmed_test_registry.csv")

    print("\nSearching PubMed...")
    pmids = search_pubmed(SEARCH_QUERY, MAX_RESULTS)

    print(f"PMIDs found: {len(pmids)}")
    print(pmids)

    time.sleep(1)

    print("\nFetching paper details...")
    xml_data = fetch_details(pmids)

    print("\nParsing XML...")
    df = parse_pubmed_xml(xml_data)

    df.to_csv(out_file, index=False)

    print("\nSaved file:")
    print(out_file)
    print("\nPreview:")
    print(df.head())
