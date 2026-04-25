from src.qc.depmap_qc import run_depmap_qc


if __name__ == "__main__":
    run_depmap_qc(
        raw_dir="data_sources/depmap_ccle/raw",
        out_dir="outputs/qc/depmap_ccle/DepMap_QC_v1",
    )
