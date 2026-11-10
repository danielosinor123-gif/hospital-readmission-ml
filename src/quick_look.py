"""
Quick first look at the raw dataset.

This script exists so you can always type:
    python src/quick_look.py
and re-verify the basic facts about the data before doing anything else.
Real analysis happens in the notebooks; this is just a fast sanity check.
"""
from pathlib import Path
import pandas as pd                    # pandas = tables (DataFrames) in Python

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"

# na_values tells pandas which placeholders really mean "missing".
# In this dataset "?" is the missing-value marker, and unknown gender is "Unknown/Invalid".
df = pd.read_csv(CSV, na_values=["?"])

print("shape (rows, columns):", df.shape)
print("\ncolumns:", list(df.columns))

print("\n--- dtypes ---")
print(df.dtypes.to_string())

print("\n--- missing values (only columns that have them) ---")
missing = df.isna().sum()
print(missing[missing > 0].to_string())

print("\n--- readmitted (the target candidate) ---")
print(df["readmitted"].value_counts().to_string())
print(df["readmitted"].value_counts(normalize=True).round(4).to_string())

print("\n--- first 3 rows ---")
print(df.head(3).T.to_string())

print("\n--- encounter_id unique?", df["encounter_id"].is_unique,
      "| patient_nbr unique?", df["patient_nbr"].is_unique,
      "| unique patients:", df["patient_nbr"].nunique())
print("rows per patient: max =", df["patient_nbr"].value_counts().max())
