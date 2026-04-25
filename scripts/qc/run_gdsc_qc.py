import os
import shutil
import pandas as pd

from src.qc.gdsc_qc import run_gdsc_qc, write_gdsc_qc_outputs


INPUT_PATH = "outputs/processed/gdsc/gdsc_iviv_normalized_v1.csv"
OUT_DIR = "outputs/qc/gdsc/GDSC_QC_v1"
TRAINING_READY_DIR = "outputs/training_ready/gdsc"


if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    print("Loading GDSC normalized dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print("Running GDSC source-level QC...")
    qc_df = run_gdsc_qc(df)

    print("Writing GDSC QC outputs...")
    summary = write_gdsc_qc_outputs(qc_df, OUT_DIR, input_path=INPUT_PATH)

    os.makedirs(TRAINING_READY_DIR, exist_ok=True)

    pass_src = os.path.join(OUT_DIR, "gdsc_pass_training_grade_v1.csv")
    pass_dst = os.path.join(TRAINING_READY_DIR, "gdsc_pass_training_grade_v1.csv")

    if os.path.exists(pass_src):
        shutil.copy2(pass_src, pass_dst)
        print(f"Copied PASS dataset to training-ready path: {pass_dst}")
    else:
        raise FileNotFoundError(f"PASS file not generated: {pass_src}")

    print("GDSC QC complete.")
    print(summary)
