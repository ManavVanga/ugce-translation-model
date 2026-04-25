from src.linking.gdsc_depmap_mapping import run_gdsc_depmap_mapping


if __name__ == "__main__":
    run_gdsc_depmap_mapping(
        gdsc_path="outputs/training_ready/gdsc/gdsc_pass_training_grade_v1.csv",
        depmap_model_path="outputs/processed/depmap_ccle/Model_clean.csv",
        out_dir="outputs/mapping/gdsc_depmap",
    )
