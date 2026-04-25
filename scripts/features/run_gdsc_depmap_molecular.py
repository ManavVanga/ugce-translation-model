from src.features.build_gdsc_depmap_molecular import run_gdsc_depmap_molecular


if __name__ == "__main__":
    run_gdsc_depmap_molecular(
        mapped_base_path="outputs/mapping/gdsc_depmap/GDSC_DepMap_mapped_training_base_v1.csv",
        expression_path="outputs/processed/depmap_ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes_clean.csv",
        mutation_path="outputs/processed/depmap_ccle/OmicsSomaticMutations_clean.csv",
        out_dir="outputs/training_ready/gdsc_depmap_molecular",
    )
