"""
BUILDER PART 1 of notebook 02 -- intro, imports, load processed data,
cleaning recap, feature engineering, encoding.
"""
import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)
NB_PATH = NB_DIR / "02_preprocessing_and_modeling.ipynb"

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""
# Part 2 & 3 — Preprocessing, Patient-Level Split, and Three Models
### Hospital Readmission Prediction | Diabetes 130-US Hospitals (1999–2008)

This notebook continues from `01_data_understanding.ipynb`, which produced
`data/processed/diabetes_clean_stage1.csv` (deceased rows removed, constant
columns dropped, ICD-9 groups created, binary target added).

**What happens here**
1. Finish the cleaning decisions (drop `weight`, fill the "Missing" categories).
2. Feature engineering — turn raw counts into data the model can use.
3. Encode categorical columns into numbers.
4. **Split by patient** so no patient appears in both sides.
5. Train three models: Logistic Regression, Random Forest, LightGBM.
6. Measure them properly on the untouched test pile.
"""))

# ------------------------------------------------------------------ imports
cells.append(nbf.v4.new_markdown_cell("""
## 1. Imports

| Import | Why |
|---|---|
| `numpy`, `pandas` | tables and numbers |
| `matplotlib`, `seaborn` | confusion matrices and ROC curves |
| `sklearn.model_selection.GroupShuffleSplit` | a splitter that **keeps each patient's stays together** |
| `sklearn.model_selection.StratifiedKFold` | fair cross-validation folds |
| `sklearn.metrics` | precision / recall / F1 / ROC-AUC / average precision |
| `sklearn.preprocessing.StandardScaler, OneHotEncoder` | put numbers on the same scale; turn categories into 0/1 columns |
| `sklearn.pipeline` | glue steps together without leaking test-set information |
| `LogisticRegression`, `RandomForestClassifier`, `LGBMClassifier` | our three models |
| `json`, `joblib` | save the split files and the trained models for the app |
"""))

cells.append(nbf.v4.new_code_cell("""import json
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path                     # OS-independent file paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import joblib

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["figure.dpi"] = 110

RANDOM_STATE = 42                      # fixed so results are reproducible
DATA = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC = DATA / "data" / "processed"
print("project root:", DATA)"""))

# ------------------------------------------------------------------ load
cells.append(nbf.v4.new_markdown_cell("""
## 2. Load the cleaned data from notebook 01

This file is the **output of Part 1**: 100,114 stays (deceased rows removed),
51 columns. Everything we decided in notebook 01 *without* executing is executed
here, where it belongs (in the modelling pipeline).
"""))

cells.append(nbf.v4.new_code_cell("""df = pd.read_csv(PROC / "diabetes_clean_stage1.csv", low_memory=False)
print("shape:", df.shape)
print("target rate:", round(df["readmit_30d"].mean(), 4))
df.head(3).T"""))

cells.append(nbf.v4.new_markdown_cell("""
### 2.1 Execute the documented cleaning decisions

| Decision | Code |
|---|---|
| `weight` (97% missing) — drop | `df.drop(columns=["weight"])` |
| `encounter_id` — pure row ID, no meaning | drop (but `patient_nbr` stays until the split!) |
| Fill "Missing" categories | `race`, `payer_code`, `medical_specialty` |
| Fill "Not measured" | `max_glu_serum`, `A1Cresult` |
| Diagnosis groups | already built in notebook 01; the raw codes are dropped |
"""))

cells.append(nbf.v4.new_code_cell("""# --- Execute the documented missing-value decisions ---------------------------
df = df.drop(columns=["weight"])                 # 97% missing -> unusable

df["race"] = df["race"].fillna("Missing")
df["payer_code"] = df["payer_code"].fillna("Missing")
df["medical_specialty"] = df["medical_specialty"].fillna("Missing")
df["max_glu_serum"] = df["max_glu_serum"].fillna("Not measured")
df["A1Cresult"] = df["A1Cresult"].fillna("Not measured")

# ICD-9 groups were built in notebook 01; raw codes are too granular to memorise
df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

print("Missing values left:", df.isna().sum().sum())
print("shape after cleaning:", df.shape)"""))

# ------------------------------------------------------------------ feature engineering
cells.append(nbf.v4.new_markdown_cell("""
## 3. Feature engineering

Three new features, each with a plain justification:

1. **`prior_visits_total`** = inpatient + emergency + outpatient visits.
   *"How often has this person used the health system in the past year?"* — one
   number instead of three for a tree to combine.
2. **`n_meds_active`** = how many of the 21 drugs are marked anything other than `No`.
   *"How heavily medicated is this patient?"*
3. **`n_dose_changes`** = how many drugs were `Up` or `Down` during the stay.
   *"How unstable was their treatment?"* (dose changes signal a patient being
   actively adjusted — a clinical judgement made by the ward team.)

None of these use future information: they describe history up to and during
the current stay.
"""))

cells.append(nbf.v4.new_code_cell("""DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
print("drug columns:", len(DRUG_COLS))

# 1) total health-system use in the previous year
df["prior_visits_total"] = (df["number_inpatient"]
                            + df["number_emergency"]
                            + df["number_outpatient"])

# 2) how many drugs are actually being taken (anything other than 'No')
df["n_meds_active"] = sum((df[c] != "No").astype(int) for c in DRUG_COLS)

# 3) how many drug doses were changed during the stay (Up or Down)
df["n_dose_changes"] = sum(df[c].isin(["Up", "Down"]).astype(int) for c in DRUG_COLS)

print(df[["prior_visits_total", "n_meds_active", "n_dose_changes"]].describe().round(2).to_string())"""))

cells.append(nbf.v4.new_code_cell("""# --- Do the new features actually separate the groups? -------------------------
new = df.groupby("readmit_30d")[["prior_visits_total", "n_meds_active", "n_dose_changes"]].mean()
print("Mean feature value by outcome (1 = readmitted <30d):")
print(new.round(2).to_string())
print()
print("All three are higher for readmitted patients -> they carry signal.")"""))

# ------------------------------------------------------------------ encoding
cells.append(nbf.v4.new_markdown_cell("""
## 4. Encoding — turning words into numbers

Models only do arithmetic, so every column must become a number. Three
strategies, chosen per column type:

| Column type | Strategy | Reason |
|---|---|---|
| `age` `[70-80)` | **midpoint number** (75) | the bins are ordered: 70s really are older than 30s |
| drug columns `No/Steady/Up/Down` | **ordinal 0/1/2/3** | ordered: no drug < same dose < more/less drug |
| `change`, `diabetesMed` | **0/1** | already binary |
| `race`, `medical_specialty`, ID columns, lab results, diagnosis groups | **one-hot** | no order between them, so we give each value its own 0/1 column |

One-hot example: `race = Caucasian` becomes `race_Caucasian = 1`, everything else 0.
"""))

cells.append(nbf.v4.new_code_cell("""# --- Age: bracket -> midpoint number -------------------------------------------
# "[70-80)" -> the digits are 70 and 80 -> midpoint 75
df["age_mid"] = df["age"].str.extract(r"(\\d+)").astype(int) + 5

# --- Ordered drug doses: No < Steady < Up/Down ---------------------------------
DOSE_MAP = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
for c in DRUG_COLS:
    df[c] = df[c].map(DOSE_MAP)

# --- Simple binary flags -------------------------------------------------------
df["change"] = (df["change"] == "Ch").astype(int)
df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)
df["gender"] = df["gender"].map({"Female": 0, "Male": 1, "Unknown/Invalid": 2})

print("Encoded examples:")
print(df[["age", "age_mid", "metformin", "change", "diabetesMed", "gender"]].head(5).to_string())"""))

cells.append(nbf.v4.new_code_cell("""# --- One-hot the remaining categoricals ----------------------------------------
# The three ID columns are numbers, but they are really category labels
# (28 = "psychiatric hospital"), so pandas must not treat them as quantities.
ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
for c in ID_COLS:
    df[c] = df[c].astype(str)

ONE_HOT_COLS = (["race", "payer_code", "medical_specialty", "max_glu_serum",
                 "A1Cresult", "diag_1_grp", "diag_2_grp", "diag_3_grp"]
                + ID_COLS)

df_encoded = pd.get_dummies(df, columns=ONE_HOT_COLS, dtype="int8")
print("shape after one-hot:", df_encoded.shape)
print("columns added:", df_encoded.shape[1] - df.shape[1])"""))

cells.append(nbf.v4.new_markdown_cell("""
**Why so many columns?** `medical_specialty` alone has ~70 values, so it becomes
~70 tiny 0/1 columns. That is normal and fine for tree models — each tree just
looks at one column at a time. It costs a little memory and is perfectly honest.
"""))

cells.append(nbf.v4.new_code_cell("""# --- Final feature list ----------------------------------------------------------
df_encoded = df_encoded.drop(columns=["encounter_id", "age"])   # row ID + old bracket

# patient_nbr is needed for the split but must NEVER be a feature:
# an ID number tells the model nothing medical and invites memorisation.
patients = df_encoded.pop("patient_nbr")

TARGET = "readmit_30d"
FEATURES = [c for c in df_encoded.columns if c != TARGET]

X = df_encoded[FEATURES]
y = df_encoded[TARGET]

print("X:", X.shape, "| y:", y.shape, "| unique patients:", patients.nunique())
print("positive rate:", round(y.mean(), 4))
print("sample features:", FEATURES[:8])"""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
nbf.write(nb, NB_PATH)
print("wrote", NB_PATH, "with", len(cells), "cells")
