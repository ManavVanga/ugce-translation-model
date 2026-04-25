import os
import json
import hashlib
from datetime import datetime

import pandas as pd


VERSION = "GDSC_DepMap_Molecular_v1"

MAPPED_BASE_PATH = "outputs/mapping/gdsc_depmap/GDSC_DepMap_mapped_training_base_v1.csv"
EXPRESSION_PATH = "outputs/processed/depmap_ccle/OmicsExpressionTPMLogp1HumanProteinCodingGenes_clean.csv"
MUTATION_PATH = "outputs/processed/depmap_ccle/OmicsSomaticMutations_clean.csv"
OUT_DIR = "outputs/training_ready/gdsc_depmap_molecular"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_mutation_summary(mutation_path):
    mut = pd.read_csv(mutation_path, low_memory=False)

    model_col = find_col(mut, ["ModelID", "DepMap_ID", "DepMapID"])
    gene_col = find_col(mut, ["HugoSymbol", "Hugo_Symbol", "Gene", "gene"])
    variant_col = find_col(mut, ["VariantInfo", "ProteinChange", "OncotatorVariantClassification", "Variant_Classification"])

    if model_col is None:
        raise ValueError("Mutation file missing ModelID column.")

    if gene_col is None:
        raise ValueError("Mutation file missing gene column.")

    mut["_is_mutation_row"] = 1

    summary = (
        mut.groupby(model_col)
        .agg(
            mutation_total_count=("_is_mutation_row", "sum"),
            mutation_unique_gene_count=(gene_col, "nunique"),
        )
        .reset_index()
        .rename(columns={model_col: "depmap_model_id"})
    )

    # Common cancer driver genes / pathway-level compact features
    driver_genes = [
        "TP53", "KRAS", "NRAS", "BRAF", "PIK3CA", "PTEN", "APC", "EGFR",
        "ERBB2", "ALK", "MET", "MYC", "CDKN2A", "RB1", "SMAD4", "BRCA1", "BRCA2"
    ]

    gene_upper = mut[gene_col].astype(str).str.upper()
    mut["_gene_upper"] = gene_upper

    for g in driver_genes:
        tmp = (
            mut.assign(_has_gene=(mut["_gene_upper"] == g).astype(int))
            .groupby(model_col)["_has_gene"]
            .max()
            .reset_index()
            .rename(columns={model_col: "depmap_model_id", "_has_gene": f"mut_{g}"})
        )
        summary = summary.merge(tmp, on="depmap_model_id", how="left")

    driver_cols = [f"mut_{g}" for g in driver_genes]
    summary[driver_cols] = summary[driver_cols].fillna(0).astype(int)

    return summary


def run_gdsc_depmap_molecular(
    mapped_base_path=MAPPED_BASE_PATH,
    expression_path=EXPRESSION_PATH,
    mutation_path=MUTATION_PATH,
    out_dir=OUT_DIR,
):
    ensure_dir(out_dir)

    for p in [mapped_base_path, expression_path, mutation_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    print("Loading mapped GDSC-DepMap base...")
    base = pd.read_csv(mapped_base_path, low_memory=False)

    if "depmap_model_id" not in base.columns:
        raise ValueError("Mapped base missing depmap_model_id column.")

    input_rows = len(base)

    print("Loading expression matrix...")
    expr = pd.read_csv(expression_path, low_memory=False)

    if "ModelID" not in expr.columns:
        raise ValueError("Expression file missing ModelID column.")

    expr = expr.rename(columns={"ModelID": "depmap_model_id"})

    # Safety: expression should have one row per model after cleaning.
    if expr["depmap_model_id"].duplicated().any():
        dup_count = int(expr["depmap_model_id"].duplicated(keep=False).sum())
        raise ValueError(f"Expression still contains duplicate depmap_model_id rows: {dup_count}")

    print("Joining expression features...")
    merged = base.merge(expr, on="depmap_model_id", how="left", validate="many_to_one")

    if len(merged) != input_rows:
        raise ValueError(f"Expression join changed row count: before={input_rows}, after={len(merged)}")

    expression_feature_cols = [c for c in expr.columns if c != "depmap_model_id"]

    expression_missing_rows = int(merged[expression_feature_cols].isna().all(axis=1).sum())
    expression_coverage_rows = input_rows - expression_missing_rows

    print("Building mutation summary features...")
    mut_summary = build_mutation_summary(mutation_path)

    if mut_summary["depmap_model_id"].duplicated().any():
        dup_count = int(mut_summary["depmap_model_id"].duplicated(keep=False).sum())
        raise ValueError(f"Mutation summary has duplicate depmap_model_id rows: {dup_count}")

    print("Joining mutation summary features...")
    merged = merged.merge(mut_summary, on="depmap_model_id", how="left", validate="many_to_one")

    if len(merged) != input_rows:
        raise ValueError(f"Mutation join changed row count: before={input_rows}, after={len(merged)}")

    mutation_feature_cols = [c for c in mut_summary.columns if c != "depmap_model_id"]

    # Fill mutation absence as 0 because no mutation rows for a model means no observed mutation in this source.
    merged[mutation_feature_cols] = merged[mutation_feature_cols].fillna(0)

    mutation_coverage_rows = int((merged["mutation_total_count"] > 0).sum())
    mutation_missing_rows = input_rows - mutation_coverage_rows

    # Final QC flags
    merged["molecular_feature_status"] = "PASS"

    merged.loc[
        merged[expression_feature_cols].isna().all(axis=1),
        "molecular_feature_status"
    ] = "REVIEW_MISSING_EXPRESSION"

    # Training-ready strict set: require expression coverage.
    training_ready = merged[merged["molecular_feature_status"] == "PASS"].copy()
    review = merged[merged["molecular_feature_status"] != "PASS"].copy()

    full_path = os.path.join(out_dir, "gdsc_depmap_molecular_full_v1.csv")
    train_path = os.path.join(out_dir, "gdsc_depmap_molecular_training_ready_v1.csv")
    review_path = os.path.join(out_dir, "gdsc_depmap_molecular_review_v1.csv")
    summary_path = os.path.join(out_dir, "gdsc_depmap_molecular_qc_summary_v1.json")

    merged.to_csv(full_path, index=False)
    training_ready.to_csv(train_path, index=False)
    review.to_csv(review_path, index=False)

    summary = {
        "version": VERSION,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "mapped_base_path": mapped_base_path,
            "mapped_base_sha256": sha256_file(mapped_base_path),
            "expression_path": expression_path,
            "expression_sha256": sha256_file(expression_path),
            "mutation_path": mutation_path,
            "mutation_sha256": sha256_file(mutation_path),
        },
        "counts": {
            "mapped_base_rows": int(input_rows),
            "expression_feature_count": int(len(expression_feature_cols)),
            "mutation_feature_count": int(len(mutation_feature_cols)),
            "expression_coverage_rows": int(expression_coverage_rows),
            "expression_missing_rows": int(expression_missing_rows),
            "expression_coverage_percent": round((expression_coverage_rows / input_rows) * 100, 2) if input_rows else 0,
            "mutation_coverage_rows": int(mutation_coverage_rows),
            "mutation_missing_rows": int(mutation_missing_rows),
            "mutation_coverage_percent": round((mutation_coverage_rows / input_rows) * 100, 2) if input_rows else 0,
            "training_ready_rows": int(len(training_ready)),
            "review_rows": int(len(review)),
            "training_ready_percent": round((len(training_ready) / input_rows) * 100, 2) if input_rows else 0,
        },
        "quality_gates": {
            "row_count_preserved_after_expression_join": True,
            "row_count_preserved_after_mutation_join": True,
            "training_requires_expression": True,
            "mutation_missing_filled_as_zero": True,
        },
        "policy": {
            "expression_policy": "Expression features are required for molecular training-ready PASS.",
            "mutation_policy": "Mutation summary features are added; missing mutation observations are filled as 0.",
            "cnv_crispr_policy": "CNV and CRISPR are not included in v1 molecular dataset; reserved for later feature expansion.",
        },
        "outputs": {
            "full": full_path,
            "training_ready": train_path,
            "review": review_path,
            "summary": summary_path,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_gdsc_depmap_molecular()
