import os
import pandas as pd
from src.utils.config import get_base_output_dir
from src.extraction.feature_extractor import build_features


if __name__ == "__main__":
    base_dir = get_base_output_dir()

    pass_path = os.path.join(
        base_dir,
        "curated",
        "translation_dataset_pass_v1.csv"
    )

    if not os.path.exists(pass_path):
        raise FileNotFoundError(f"PASS dataset not found: {pass_path}")

    print("Loading PASS dataset...")
    df = pd.read_csv(pass_path)

    print("Building biologically richer features...")
    feature_df = build_features(df)

    output_path = os.path.join(
        base_dir,
        "curated",
        "translation_features_v1.csv"
    )

    feature_df.to_csv(output_path, index=False)

    print("Feature extraction complete")
    print(output_path)
    print("\nPreview:")
    print(feature_df.head())
