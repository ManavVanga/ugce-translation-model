import os
import pandas as pd
from src.utils.config import get_base_output_dir
from src.extraction.row_builder import build_translation_rows
from src.extraction.normalizers import normalize_translation_rows


if __name__ == "__main__":
    base_dir = get_base_output_dir()

    extraction_path = os.path.join(
        base_dir,
        "collection",
        "literature_extraction",
        "selected_paper_extraction_v1.csv"
    )

    if not os.path.exists(extraction_path):
        raise FileNotFoundError(f"Selected extraction file not found: {extraction_path}")

    print("Loading selected paper extraction file...")
    extract_df = pd.read_csv(extraction_path)

    print("Building multi-row biological translation records...")
    row_df = build_translation_rows(extract_df)

    print("Rows generated:", len(row_df))

    if len(row_df) > 0:
        print("Applying normalization layer...")
        row_df = normalize_translation_rows(row_df)

    out_dir = os.path.join(base_dir, "collection")
    os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, "translation_row_candidates_v1.csv")
    row_df.to_csv(output_path, index=False)

    print("Saved normalized row candidates:", output_path)
    print(row_df.head())
