"""
Pattern-hunting script for Part 1: standing up the dataset.

Goal: answer "which features look related to 30-day readmission?"
Technique: split the data by the target (readmitted <30 days vs not) and
compare the DISTRIBUTIONS of every column. Useful patterns show up as
"the readmitted group is clearly different".

Run:  python src/01_patterns.py
"""
from pathlib import Path
import pandas as pd

pd.set_option("display.width", 120)     # don't wrap output lines too early

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"

# low_memory=True default warns because payer_code mixes types -> use low_memory=False
df = pd.read_csv(CSV, na_values=["?"], low_memory=False)

# ---------- 0. Basic facts (Part 1 Q1-Q3, Q5) ----------
print("=" * 70)
print("0. BASIC FACTS")
print("=" * 70)
print("shape:", df.shape)
print("unique encounters:", df["encounter_id"].nunique(),
      "| unique patients:", df["patient_nbr"].nunique())
print("exact duplicate rows:", df.duplicated().sum())

# Target: the original 'readmitted' column has 3 values.
print("\nreadmitted value counts:")
print(df["readmitted"].value_counts().to_string())

# Our engineering choice: binary target = readmitted within 30 days
df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)
print("\npositives (1 = readmitted <30d):", df["readmit_30d"].sum(),
      f"= {df['readmit_30d'].mean():.2%} of all encounters")

# ---------- 1. Patterns: how each column relates to the target ----------
print("\n" + "=" * 70)
print("1. READMISSION RATE BY CATEGORICAL COLUMN")
print("=" * 70)

for col in ["age", "gender", "race"]:
    g = df.groupby(col, observed=True)["readmit_30d"].agg(n="count", rate="mean")
    print(f"\n--- {col} ---")
    print(g.round(4).to_string())

print("\n--- admission_type_id / discharge_disposition_id / admission_source_id ---")
for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
    g = df.groupby(col, observed=True)["readmit_30d"].agg(n="count", rate="mean")
    g = g[g["n"] >= 100].sort_values("n", ascending=False)   # keep only common IDs
    print(f"\n--- {col} (only IDs with >=100 encounters) ---")
    print(g.round(4).to_string())

print("\n--- service utilisation: mean per encounter, by target ---")
numeric_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                "num_medications", "number_outpatient", "number_emergency",
                "number_inpatient", "number_diagnoses"]
print(df.groupby("readmit_30d")[numeric_cols].mean().round(2).to_string())

print("\n--- readmission rate by number_inpatient (count of past stays) ---")
g = df.groupby("number_inpatient", observed=True)["readmit_30d"].agg(n="count", rate="mean")
print(g.round(4).to_string())

print("\n--- medications / diabetes flags ---")
for col in ["diabetesMed", "change", "insulin", "metformin", "A1Cresult", "max_glu_serum"]:
    g = df.groupby(col, observed=True, dropna=False)["readmit_30d"].agg(n="count", rate="mean")
    print(f"\n--- {col} ---")
    print(g.round(4).to_string())

# ---------- 2. Diagnosis codes (Part 1 Q4) ----------
print("\n" + "=" * 70)
print("2. DIAGNOSIS CODES (diag_1/2/3 are ICD-9 codes)")
print("=" * 70)
print("sample diag_1 values:", df["diag_1"].dropna().unique()[:15])

def icd9_group(code):
    """Map an ICD-9 code to a coarse clinical group (common shortcut in papers)."""
    if isinstance(code, float):                 # NaN
        return "missing"
    s = str(code)
    if s[0] in ("V", "E"):                      # V-codes = non-disease factors
        return "other(V/E)"
    try:
        n = float(s.split(".")[0])
    except ValueError:
        return "missing"
    if 250 <= n < 251:
        return "diabetes(250)"
    if 390 <= n < 460 or n == 785:
        return "circulatory"
    if 460 <= n < 520 or n == 786:
        return "respiratory"
    if 520 <= n < 580 or n == 787:
        return "digestive"
    if 800 <= n < 1000:
        return "injury"
    if 710 <= n < 740:
        return "musculoskeletal"
    if 580 <= n < 630 or n == 788:
        return "genitourinary"
    if 140 <= n < 240:
        return "neoplasms"
    return "other"

for col in ["diag_1", "diag_2", "diag_3"]:
    df[f"{col}_grp"] = df[col].map(icd9_group)
    g = df.groupby(f"{col}_grp", observed=True)["readmit_30d"].agg(n="count", rate="mean")
    print(f"\n--- {col} grouped ---")
    print(g.sort_values("rate", ascending=False).round(4).to_string())

# ---------- 3. Data-quality tails ----------
print("\n" + "=" * 70)
print("3. DATA QUALITY DETAILS")
print("=" * 70)
print("gender counts:")
print(df["gender"].value_counts(dropna=False).to_string())
print("\nweight: present in", df["weight"].notna().sum(), "rows of", len(df))
print("max_glu_serum:", df["max_glu_serum"].value_counts(dropna=False).to_dict())
print("A1Cresult:", df["A1Cresult"].value_counts(dropna=False).to_dict())
print("\nrows per patient (top 5):")
print(df["patient_nbr"].value_counts().head(5).to_string())
print("\nfraction of patients with >1 encounter:",
      f"{(df['patient_nbr'].value_counts() > 1).mean():.2%}")
