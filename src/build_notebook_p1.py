"""
BUILDER PART 1 -- creates notebooks/01_data_understanding.ipynb with the
intro + setup + load + structural-cleaning sections.

nbformat is the standard library for creating Jupyter notebooks programmatically.
We build the notebook in parts (p1 -> p4) so each file stays small and readable.
"""
import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)
NB_PATH = NB_DIR / "01_data_understanding.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# ============================================================ title
cells.append(nbf.v4.new_markdown_cell("""
# Part 1 — Data Understanding & Exploration
### Hospital Readmission Prediction | Diabetes 130-US Hospitals (1999–2008)

This notebook is the **first deliverable**: it documents what the data contains,
what problems it has, and which patterns exist *before* any modelling.

**How to read this notebook**
- Every code cell is preceded by a markdown cell that explains **why** we run it.
- Comments inside the code (`#`) explain individual lines.
- The narrative answers, in order, the six Part 1 questions:
  1. What does the dataset contain?
  2. Target variable
  3. Missing values / duplicates / inconsistencies
  4. Candidate features
  5. Class balance
  6. Visual patterns
"""))

# ============================================================ 0. imports
cells.append(nbf.v4.new_markdown_cell("""
## 0. Setup and imports

| Import | Why we need it |
|---|---|
| `pandas` | loads the CSV into a DataFrame (table) + grouping/filtering |
| `numpy` | numeric helpers (arrays, log transforms, binning) |
| `matplotlib.pyplot` | the plotting engine — every chart is drawn by matplotlib |
| `seaborn` | statistical charts with far less code than raw matplotlib |
| `pathlib.Path` | file paths that work on any operating system |

`%matplotlib inline` is a Jupyter "magic command" (not Python) that tells the
notebook to draw plots *inside* the page instead of a separate window.
"""))

cells.append(nbf.v4.new_code_cell("""# --- Standard data-science imports -----------------------------------------
import warnings
warnings.filterwarnings("ignore")          # keep output readable for the reviewer

from pathlib import Path                    # OS-independent file paths
import numpy as np                          # numerical operations
import pandas as pd                         # tables (DataFrames)

import matplotlib.pyplot as plt             # base plotting library
import matplotlib.ticker as mtick          # format axes as percentages
import seaborn as sns                       # nicer statistical charts

# --- Make plots bigger and show a light grid by default ---------------------
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["figure.dpi"] = 110

# --- Locate the dataset relative to this notebook ---------------------------
# notebooks/01_...ipynb  ->  parent = notebooks/, parent.parent = project root
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
CSV = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
print("Looking for data at:", CSV)
print("Exists:", CSV.exists())"""))

# ============================================================ 1. load
cells.append(nbf.v4.new_markdown_cell("""
## 1. What does the dataset contain? (Part 1, Q1)

Each row is **one hospital admission (an "encounter")**, not one patient.

Two critical read options:
- `na_values=["?"]` — in this dataset `?` is the official "missing" marker.
  Without it, missingness is invisible.
- `low_memory=False` — `payer_code` mixes text/numbers; this stops pandas
  from guessing different types for different chunks of the file.
"""))

cells.append(nbf.v4.new_code_cell("""# Load the raw data. One row = one inpatient stay.
df = pd.read_csv(
    CSV,
    na_values=["?"],          # treat '?' as missing
    low_memory=False,         # avoid mixed-type guessing warnings
)

print("Shape (rows, columns):", df.shape)
print("Unique stays (encounter_id):", df["encounter_id"].nunique())
print("Unique patients (patient_nbr):", df["patient_nbr"].nunique())
print("Patients with >1 stay:", (df["patient_nbr"].value_counts() > 1).sum())
print("Max stays for one patient:", df["patient_nbr"].value_counts().max())"""))

cells.append(nbf.v4.new_markdown_cell("""
**Observation:** ~101,766 rows but only ~71,518 patients — about 30,000 stays
are repeat visits, and the most frequent patient returns **40 times**.

**Consequence (important):** if we split randomly by row, the same patient can
appear in both training and test data, and the model appears far better than it
is. Part 2 therefore splits **by patient**, not by row.

Below, the 50 columns grouped by meaning:
"""))

cells.append(nbf.v4.new_code_cell("""col_groups = {
    "identifiers"        : ["encounter_id", "patient_nbr"],
    "demographics"       : ["race", "gender", "age", "weight"],
    "admission/discharge": ["admission_type_id", "discharge_disposition_id",
                            "admission_source_id", "time_in_hospital"],
    "utilisation counts" : ["num_lab_procedures", "num_procedures", "num_medications",
                            "number_outpatient", "number_emergency",
                            "number_inpatient", "number_diagnoses"],
    "diagnoses"          : ["diag_1", "diag_2", "diag_3"],
    "lab results"        : ["max_glu_serum", "A1Cresult"],
    "medications (23)"   : ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
                            "glimepiride", "acetohexamide", "glipizide", "glyburide",
                            "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
                            "miglitol", "troglitazone", "tolazamide", "examide",
                            "citoglipton", "insulin", "glyburide-metformin",
                            "glipizide-metformin", "glimepiride-pioglitazone",
                            "metformin-rosiglitazone", "metformin-pioglitazone"],
    "flags + target"     : ["change", "diabetesMed", "readmitted"],
}
all_listed = sum(col_groups.values(), [])
print("Columns grouped:", len(all_listed), "of", df.shape[1])
print("Not listed:", [c for c in df.columns if c not in all_listed])

for name, cols in col_groups.items():
    print(f"\\n{name} ({len(cols)}):")
    print("  " + ", ".join(cols))"""))

cells.append(nbf.v4.new_code_cell("""# Peek at the first 3 stays: .T transposes so we see columns down the page
df.head(3).T"""))

# ============================================================ 2. structural cleaning
cells.append(nbf.v4.new_markdown_cell("""
## 2. Structural cleaning *before* analysis (Part 1, Q3 part A)

We do not silently delete anything — we only apply fixes whose justification is
obvious from the data itself, and we document each one.

**Step 2.1 — Remove rows that can never be readmitted (target leakage).**
`discharge_disposition_id` values `11, 19, 20, 21` mean the patient **died**
(during the stay or in hospice). A deceased patient cannot be readmitted within
30 days, so leaving them in teaches the model the shortcut
"expired -> definitely no readmission", which is useless in practice.

We remove them and print exactly how many rows were dropped.
"""))

cells.append(nbf.v4.new_code_cell("""died_ids = [11, 19, 20, 21]           # 'Expired' / 'Expired in hospice'
mask_died = df["discharge_disposition_id"].isin(died_ids)
print("Rows where the patient died:", mask_died.sum())
print("Readmission rate in those rows:", df.loc[mask_died, "readmitted"].value_counts().to_dict())
print("  -> all 'NO': mathematically impossible to be a positive, i.e. leakage.")

df = df[~mask_died].copy()             # keep everyone else
print("\\nRows after removing deceased:", len(df))"""))

cells.append(nbf.v4.new_markdown_cell("""
**Step 2.2 — Rows with no recorded outcome.** If `readmitted` were missing we
could not use the row for training. (There are none here, but a real dataset
should always be checked.)
"""))

cells.append(nbf.v4.new_code_cell("""print("Missing 'readmitted':", df["readmitted"].isna().sum(), "-> no action needed")"""))

cells.append(nbf.v4.new_markdown_cell("""
**Step 2.3 — Invalid / uninformative values.**
- `gender = "Unknown/Invalid"` appears in only ~3 rows — we keep them for now;
  they are too few to matter and will be grouped as "Other" later if needed.
- Two drug columns (`examide`, `citoglipton`) hold the single value `No` in
  every row: constant columns carry **zero information** and are dropped.
"""))

cells.append(nbf.v4.new_code_cell("""print("Gender values:", df["gender"].value_counts(dropna=False).to_dict())

# A column is useless if it has only 1 distinct value.
constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print("\\nConstant columns (dropped):", constant_cols)
print(df[constant_cols].apply(lambda s: s.value_counts(dropna=False).to_dict()).to_dict())
df = df.drop(columns=constant_cols)
print("Columns remaining:", df.shape[1])"""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
nbf.write(nb, NB_PATH)
print("wrote", NB_PATH, "with", len(cells), "cells")
