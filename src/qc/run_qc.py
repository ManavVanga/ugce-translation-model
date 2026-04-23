import os
import json
import pandas as pd
from src.utils.config import get_base_output_dir
from src.qc.qc_engine import run_full_qc
from src.qc.router import route_by_qc


if __name__ == "__main__":
    base_dir = get_base_output_dir()

    candidate_path = os.path.join(
        base_dir,
        "collection",
        "translation_row_candidates_v1.csv"
    )

    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Row candidate file not found: {candidate_path}")

    print("Loading row candidates...")
    df = pd.read_csv(candidate_path)

    print("Running locked-schema QC...")
    df_qc = run_full_qc(df)

    curated_dir = os.path.join(base_dir, "curated")
    os.makedirs(curated_dir, exist_ok=True)

    qc_path = os.path.join(curated_dir, "translation_dataset_qc_v1.csv")
    summary_path = os.path.join(curated_dir, "translation_dataset_qc_summary_v1.json")

    df_qc.to_csv(qc_path, index=False)

    summary = {
        "total_rows": int(len(df_qc)),
        "pass_rows": int((df_qc["qc_overall_status"] == "PASS").sum()),
        "review_rows": int((df_qc["qc_overall_status"] == "REVIEW").sum()),
        "fail_rows": int((df_qc["qc_overall_status"] == "FAIL").sum()),
        "schema_pass": int((df_qc["qc_schema_pass"] == "PASS").sum()),
        "format_pass": int((df_qc["qc_format_pass"] == "PASS").sum()),
        "biological_pass": int((df_qc["qc_biological_pass"] == "PASS").sum()),
        "linkage_pass": int((df_qc["qc_linkage_pass"] == "PASS").sum()),
        "evidence_pass": int((df_qc["qc_evidence_pass"] == "PASS").sum())
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Routing PASS / REVIEW / FAIL...")
    route_by_qc(df_qc, base_dir)

    print("QC COMPLETE")
    print("QC file:", qc_path)
    print("Summary file:", summary_path)
    print("Summary:", summary)
