import os
import pandas as pd

INPUT = "data_sources/gdsc/raw/gdsc_drug_response.csv"
OUT = "outputs/processed/gdsc/gdsc_iviv_normalized_v1.csv"

os.makedirs("outputs/processed/gdsc", exist_ok=True)

df = pd.read_csv(INPUT)

# TEMP basic mapping (we refine later)
df["drug_name_standard"] = df.get("drug_name", "")
df["vitro_system_class"] = "cell_line"
df["assay_type"] = "viability"
df["assay_endpoint"] = "IC50"
df["response_value_standard"] = df.get("ic50", "")
df["response_metric_standard"] = "IC50_uM"
df["outcome_level"] = "in_vitro"

df.to_csv(OUT, index=False)

print("Normalized GDSC saved:", OUT)
