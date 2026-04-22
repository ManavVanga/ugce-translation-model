import requests
import json
import time
from src.utils.config import load_drive_paths

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SEARCH_QUERY = "(organoid OR organ-on-chip OR microphysiological OR 2D culture OR CRISPR) AND (drug response OR IC50 OR viability OR toxicity)"
MAX_RESULTS = 5   # keep small for testing


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


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Loading config...")
    paths = load_drive_paths()

    print("\nSearching PubMed...")
    pmids = search_pubmed(SEARCH_QUERY, MAX_RESULTS)

    print(f"PMIDs found: {len(pmids)}")
    print(pmids)

    time.sleep(1)

    print("\nFetching paper details...")
    xml_data = fetch_details(pmids)

    print("\nSample XML output (first 1000 chars):")
    print(xml_data[:1000])
