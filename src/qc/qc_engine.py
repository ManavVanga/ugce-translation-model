import pandas as pd


def qc_schema_check(df):
    required_cols = [
        "paper_id", "pmid", "title", "abstract", "year",
        "candidate_class", "priority_score", "priority_bucket"
    ]

    results = []
    for _, row in df.iterrows():
        missing = [col for col in required_cols if pd.isna(row.get(col))]
        results.append("PASS" if len(missing) == 0 else "FAIL")

    df["qc_schema_pass"] = results
    return df


def qc_format_check(df):
    results = []

    for _, row in df.iterrows():
        try:
            year = int(row["year"]) if str(row["year"]).isdigit() else None
            if year and 1900 <= year <= 2100:
                results.append("PASS")
            else:
                results.append("FAIL")
        except:
            results.append("FAIL")

    df["qc_format_pass"] = results
    return df


def qc_biological_check(df):
    results = []

    for _, row in df.iterrows():
        text = f"{row.get('title','')} {row.get('abstract','')}".lower()

        if any(k in text for k in ["drug", "treatment", "response", "toxicity"]):
            results.append("PASS")
        else:
            results.append("FAIL")

    df["qc_biological_pass"] = results
    return df


def qc_linkage_check(df):
    results = []

    for _, row in df.iterrows():
        if row["candidate_class"] == "combined_candidate":
            results.append("PASS")
        else:
            results.append("FAIL")

    df["qc_linkage_pass"] = results
    return df


def qc_evidence_check(df):
    results = []

    for _, row in df.iterrows():
        score = row.get("priority_score", 0)
        if score >= 4:
            results.append("PASS")
        else:
            results.append("FAIL")

    df["qc_evidence_pass"] = results
    return df


def final_qc_status(df):
    overall = []

    for _, row in df.iterrows():
        checks = [
            row["qc_schema_pass"],
            row["qc_format_pass"],
            row["qc_biological_pass"],
            row["qc_linkage_pass"],
            row["qc_evidence_pass"]
        ]

        if all(c == "PASS" for c in checks):
            overall.append("PASS")
        else:
            overall.append("FAIL")

    df["qc_overall_status"] = overall
    return df


def run_full_qc(df):
    df = qc_schema_check(df)
    df = qc_format_check(df)
    df = qc_biological_check(df)
    df = qc_linkage_check(df)
    df = qc_evidence_check(df)
    df = final_qc_status(df)

    return df
