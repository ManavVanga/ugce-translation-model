import os
import pandas as pd

from src.qc.gdsc_qc import run_gdsc_qc, write_gdsc_qc_outputs


INPUT_PATH = "data/processed/gdsc/gdsc_iviv_normalized_v1.csv"
OUT_DIR = "data/qc/gdsc/GDSC_QC_v1"


if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    print("Loading GDSC normalized dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print("Running GDSC source-level QC...")
    qc_df = run_gdsc_qc(df)

    print("Writing PASS / REVIEW / FAIL outputs...")
    write_gdsc_qc_outputs(qc_df, OUT_DIR, input_path=INPUT_PATH)
