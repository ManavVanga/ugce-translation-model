import os
import pandas as pd
from src.utils.config import get_base_output_dir
from src.qc.qc_engine import run_full_qc
from src.qc.router import route_by_qc


if __name__ == "__main__":
    base_dir = get_base_output_dir()

    registry_path = os.path.join(
        base_dir,
        "collection",
        "literature_registry",
        "pubmed_paper_registry_v1.csv"
    )

    if not os.path.exists(registry_path):
        raise FileNotFoundError("Registry file not found")

    print("Loading registry...")
    df = pd.read_csv(registry_path)

    print("Running QC...")
    df_qc = run_full_qc(df)

    qc_path = os.path.join(
        base_dir,
        "curated",
        "translation_dataset_qc_v1.csv"
    )

    os.makedirs(os.path.dirname(qc_path), exist_ok=True)
    df_qc.to_csv(qc_path, index=False)

    print("Routing PASS / FAIL...")
    route_by_qc(df_qc, base_dir)

    print("QC COMPLETE")
