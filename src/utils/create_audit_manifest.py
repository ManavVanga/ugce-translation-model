import os
from datetime import datetime
from src.utils.config import get_base_output_dir
from src.utils.manifest import build_manifest


if __name__ == "__main__":
    base_dir = get_base_output_dir()

    runs_dir = os.path.join(base_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    run_name = f"audit_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tracked_files = [
        os.path.join(base_dir, "collection", "literature_registry", "pubmed_paper_registry_v1.csv"),
        os.path.join(base_dir, "collection", "literature_registry", "pubmed_paper_screening_v1.csv"),
        os.path.join(base_dir, "collection", "literature_extraction", "selected_paper_extraction_v1.csv"),
        os.path.join(base_dir, "collection", "translation_row_candidates_v1.csv"),
        os.path.join(base_dir, "collection", "translation_row_master_v1.csv"),
        os.path.join(base_dir, "collection", "translation_row_duplicates_v1.csv"),
        os.path.join(base_dir, "collection", "translation_row_append_summary_v1.json"),
        os.path.join(base_dir, "curated", "translation_dataset_qc_v1.csv"),
        os.path.join(base_dir, "curated", "translation_dataset_pass_v1.csv"),
        os.path.join(base_dir, "curated", "translation_dataset_review_v1.csv"),
        os.path.join(base_dir, "curated", "translation_dataset_fail_v1.csv"),
        os.path.join(base_dir, "curated", "translation_dataset_qc_summary_v1.json"),
        os.path.join(base_dir, "curated", "translation_features_v1.csv"),
    ]

    manifest_path = os.path.join(run_dir, "run_manifest_v1.json")
    manifest = build_manifest(tracked_files, run_name, manifest_path)

    print("Manifest created:")
    print(manifest_path)
    print(manifest)
