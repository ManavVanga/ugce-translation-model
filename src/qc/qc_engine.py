import pandas as pd
import numpy as np


MANDATORY_FIELDS = [
    "drug_name_standard",
    "intervention_type",
    "vitro_system_class",
    "tissue_context",
    "assay_type",
    "assay_endpoint",
    "dose_normalized_uM",
    "exposure_time_hours",
    "response_value_standard",
    "response_metric_standard",
    "outcome_level",
]

ALLOWED_OUTCOME_LEVELS = {"in_vitro", "animal_in_vivo", "human_clinical"}
ALLOWED_EFFECT_DIRECTIONS = {"decrease", "increase", "none", ""}
ALLOWED_EVIDENCE_WEIGHTS = {"high", "medium", "low", ""}


def is_missing(x):
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}

def safe_float(x):
    try:
        if is_missing(x):
            return np.nan
        return float(x)
    except:
        return np.nan

def qc_schema_check(df):
    results = []
    reasons = []

    for _, row in df.iterrows():
        missing_required = [c for c in MANDATORY_FIELDS if is_missing(row.get(c, ""))]
        has_outcome = (not is_missing(row.get("invivo_outcome_label", ""))) or (not is_missing(row.get("human_outcome_label", "")))
        outcome_level = str(row.get("outcome_level", "")).strip()

        local_reasons = []

        if missing_required:
            local_reasons.append(f"missing_required={missing_required}")
        if not has_outcome:
            local_reasons.append("missing_outcome_label")
        if outcome_level not in ALLOWED_OUTCOME_LEVELS:
            local_reasons.append("invalid_outcome_level")

        if local_reasons:
            results.append("FAIL")
            reasons.append("; ".join(local_reasons))
        else:
            results.append("PASS")
            reasons.append("schema_ok")

    df["qc_schema_pass"] = results
    df["qc_schema_reason"] = reasons
    return df

def qc_format_check(df):
    results = []
    reasons = []

    for _, row in df.iterrows():
        local_reasons = []

        dose = safe_float(row.get("dose_normalized_uM", np.nan))
        exposure = safe_float(row.get("exposure_time_hours", np.nan))
        response = safe_float(row.get("response_value_standard", np.nan))

        if np.isnan(dose):
            local_reasons.append("dose_not_numeric")
        elif dose <= 0:
            local_reasons.append("dose_nonpositive")

        if np.isnan(exposure):
            local_reasons.append("time_not_numeric")
        elif exposure <= 0:
            local_reasons.append("time_nonpositive")

        if np.isnan(response):
            local_reasons.append("response_not_numeric")

        if local_reasons:
            results.append("FAIL")
            reasons.append("; ".join(local_reasons))
        else:
            results.append("PASS")
            reasons.append("format_ok")

    df["qc_format_pass"] = results
    df["qc_format_reason"] = reasons
    return df

def qc_biological_check(df):
    results = []
    reasons = []

    for _, row in df.iterrows():
        local_reasons = []

        response = safe_float(row.get("response_value_standard", np.nan))
        response_metric = str(row.get("response_metric_standard", "")).strip()
        assay_type = str(row.get("assay_type", "")).strip().lower()
        effect_direction = str(row.get("effect_direction", "")).strip().lower()

        if response_metric == "IC50_uM":
            if np.isnan(response):
                local_reasons.append("ic50_missing")
            elif not (0 < response < 10000):
                local_reasons.append("ic50_out_of_range")

        if response_metric == "viability_percent":
            if np.isnan(response):
                local_reasons.append("viability_missing")
            elif not (0 <= response <= 100):
                local_reasons.append("viability_out_of_range")

        if assay_type == "viability" and effect_direction not in ALLOWED_EFFECT_DIRECTIONS:
            local_reasons.append("invalid_effect_direction_for_viability")

        if local_reasons:
            results.append("FAIL")
            reasons.append("; ".join(local_reasons))
        else:
            results.append("PASS")
            reasons.append("biological_ok")

    df["qc_biological_pass"] = results
    df["qc_biological_reason"] = reasons
    return df

def qc_linkage_check(df):
    results = []
    reasons = []

    for _, row in df.iterrows():
        has_invitro = (
            not is_missing(row.get("vitro_system_class", "")) and
            not is_missing(row.get("assay_type", "")) and
            not is_missing(row.get("response_value_standard", ""))
        )
        has_animal = not is_missing(row.get("invivo_outcome_label", ""))
        has_human = not is_missing(row.get("human_outcome_label", ""))

        if has_invitro and (has_animal or has_human):
            results.append("PASS")
            reasons.append("translation_link_present")
        elif has_invitro:
            results.append("REVIEW")
            reasons.append("invitro_only_supporting")
        else:
            results.append("FAIL")
            reasons.append("missing_translation_link")

    df["qc_linkage_pass"] = results
    df["qc_linkage_reason"] = reasons
    return df

def qc_evidence_check(df):
    results = []
    reasons = []

    for _, row in df.iterrows():
        evidence_weight = str(row.get("evidence_weight", "")).strip().lower()
        has_animal = not is_missing(row.get("invivo_outcome_label", ""))
        has_human = not is_missing(row.get("human_outcome_label", ""))
        has_invitro = not is_missing(row.get("vitro_system_class", ""))

        if evidence_weight not in ALLOWED_EVIDENCE_WEIGHTS:
            results.append("FAIL")
            reasons.append("invalid_evidence_weight")
            continue

        if has_invitro and has_human:
            results.append("PASS")
            reasons.append("high_weight_evidence")
        elif has_invitro and has_animal:
            results.append("PASS")
            reasons.append("medium_weight_evidence")
        elif has_invitro:
            results.append("REVIEW")
            reasons.append("low_weight_supporting")
        else:
            results.append("FAIL")
            reasons.append("weak_evidence")

    df["qc_evidence_pass"] = results
    df["qc_evidence_reason"] = reasons
    return df

def final_qc_status(df):
    overall_status = []
    overall_reason = []

    for _, row in df.iterrows():
        checks = [
            row.get("qc_schema_pass", ""),
            row.get("qc_format_pass", ""),
            row.get("qc_biological_pass", ""),
            row.get("qc_linkage_pass", ""),
            row.get("qc_evidence_pass", ""),
        ]

        if "FAIL" in checks:
            overall_status.append("FAIL")
            overall_reason.append("one_or_more_fail")
        elif "REVIEW" in checks:
            overall_status.append("REVIEW")
            overall_reason.append("manual_review_needed")
        else:
            overall_status.append("PASS")
            overall_reason.append("all_checks_passed")

    df["qc_overall_status"] = overall_status
    df["qc_overall_reason"] = overall_reason
    return df

def run_full_qc(df):
    df = qc_schema_check(df)
    df = qc_format_check(df)
    df = qc_biological_check(df)
    df = qc_linkage_check(df)
    df = qc_evidence_check(df)
    df = final_qc_status(df)
    return df
