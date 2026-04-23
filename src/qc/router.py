import os
import pandas as pd


def route_by_qc(df, base_dir):
    curated_dir = os.path.join(base_dir, "curated")
    os.makedirs(curated_dir, exist_ok=True)

    pass_df = df[df["qc_overall_status"] == "PASS"].copy()
    review_df = df[df["qc_overall_status"] == "REVIEW"].copy()
    fail_df = df[df["qc_overall_status"] == "FAIL"].copy()

    pass_path = os.path.join(curated_dir, "translation_dataset_pass_v1.csv")
    review_path = os.path.join(curated_dir, "translation_dataset_review_v1.csv")
    fail_path = os.path.join(curated_dir, "translation_dataset_fail_v1.csv")

    pass_df.to_csv(pass_path, index=False)
    review_df.to_csv(review_path, index=False)
    fail_df.to_csv(fail_path, index=False)

    print(f"PASS rows: {len(pass_df)}")
    print(f"REVIEW rows: {len(review_df)}")
    print(f"FAIL rows: {len(fail_df)}")
