from src.normalize.depmap_clean import run_depmap_clean


if __name__ == "__main__":
    run_depmap_clean(
        raw_dir="data_sources/depmap_ccle/raw",
        out_dir="outputs/processed/depmap_ccle",
    )
