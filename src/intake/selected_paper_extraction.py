import os
import re
import time
import json
import requests
import pandas as pd
from datetime import datetime
from src.utils.config import get_base_output_dir

PMC_IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"

SLEEP = 0.3


def clean(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def get_pmcid_from_pmid(pmid):
    r = requests.get(
        PMC_IDCONV_URL,
        params={"ids": str(pmid), "idtype": "pmid", "format": "json"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    records = data.get("records", [])
    if not records:
        return ""
    return clean(records[0].get("pmcid", ""))


def fetch_pmc_bioc_sections(pmcid):
    if not pmcid:
        return "", "no_pmcid", ""

    pmcid_num = pmcid.replace("PMC", "")
    url = PMC_OA_URL.format(pmcid=pmcid_num)

    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return "", f"http_{r.status_code}", ""

        data = r.json()

        all_passages = []
        priority_passages = []

        priority_section_terms = [
            "result", "results",
            "method", "methods",
            "materials and methods",
            "figure", "fig.",
            "table",
            "supplement",
            "dose", "response",
        ]

        for doc in data.get("documents", []):
            for p in doc.get("passages", []):
                text = clean(p.get("text", ""))
                if not text:
                    continue

                section = ""
                infons = p.get("infons", {})
                if isinstance(infons, dict):
                    section = clean(
                        infons.get("section_type", "")
                        or infons.get("type", "")
                        or infons.get("section", "")
                    )

                block = f"[SECTION={section}] {text}"
                all_passages.append(block)

                lower_block = block.lower()
                if any(term in lower_block for term in priority_section_terms):
                    priority_passages.append(block)

        fulltext = "\n".join(all_passages)
        priority_text = "\n".join(priority_passages)

        if fulltext:
            return fulltext, "ok", priority_text

        return "", "empty", ""

    except Exception as e:
        return "", f"error_{type(e).__name__}", ""


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
        (screen_df["manual_keep_decision"].astype(str).str.strip().str.lower() == "keep_high_priority")
        & (screen_df["manual_extract_ready"].astype(str).str.strip().str.lower() == "yes")
    ].copy()

    print("Selected papers for extraction:", len(keep_df))

    if len(keep_df) == 0:
        print("No selected papers. Exiting.")
        raise SystemExit(0)

    rows = []

    for i, (_, row) in enumerate(keep_df.iterrows(), start=1):
        pmid = clean(row.get("pmid", ""))
        print(f"Extracting {i}/{len(keep_df)} | PMID={pmid}")

        pmcid = ""
        fulltext = ""
        priority_text = ""
        status = "not_attempted"

        try:
            pmcid = get_pmcid_from_pmid(pmid)
            time.sleep(SLEEP)

            fulltext, status, priority_text = fetch_pmc_bioc_sections(pmcid)
            time.sleep(SLEEP)

        except Exception as e:
            status = f"error_{type(e).__name__}"

        rows.append({
            "paper_id": clean(row.get("paper_id", "")),
            "pmid": pmid,
            "pmcid": pmcid,
            "doi": clean(row.get("doi", "")),
            "title": clean(row.get("title", "")),
            "journal": clean(row.get("journal", "")),
            "year": clean(row.get("year", "")),
            "authors": clean(row.get("authors", "")),
            "pubmed_url": clean(row.get("pubmed_url", "")),
            "source_database": "PubMed/PMC",
            "manual_true_study_type": clean(row.get("manual_true_study_type", "")),
            "manual_keep_decision": clean(row.get("manual_keep_decision", "")),
            "manual_reason": clean(row.get("manual_reason", "")),
            "abstract": clean(row.get("abstract", "")),
            "pmc_fulltext_status": status,
            "pmc_fulltext": fulltext,
            "pmc_priority_text": priority_text,
            "manual_extract_ready": clean(row.get("manual_extract_ready", "")),
            "data_added_date": datetime.now().strftime("%Y-%m-%d"),
            "curated_by": "auto_fulltext_extractor_v2"
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(extraction_file, index=False)

    print("Saved extraction file:", extraction_file)
    print(out_df[["paper_id", "pmid", "pmcid", "pmc_fulltext_status"]].head(20))

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
        "source_type": "pmc_fulltext_section_extraction",
        "study_type": "literature_extraction",
        "num_records_added": int(len(out_df)),
        "curated_by": "auto_fulltext_extractor_v2",
        "notes": "Extracted selected papers with PMC full text and priority result/method/table passages"
    }])

    log_df = pd.concat([log_df, new_log], ignore_index=True)
    log_df.to_csv(log_file, index=False)
