"""
BUILDER PART 2 of notebook 02 -- leakage audit, patient-level split,
and saving the splits for later notebooks / the app.
"""
import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = PROJECT_ROOT / "notebooks" / "02_preprocessing_and_modeling.ipynb"

nb = nbf.read(NB_PATH, as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell("""
## 5. Leakage audit — is every feature known *before* the decision moment?

The model is meant to help at **discharge time**: when the patient is about to
leave, we want to know whether to arrange follow-up care. A feature is safe only
if its value exists at that moment. Anything known only *after* discharge would
be cheating (the model would look brilliant in testing and fail in reality).

| Feature group | Known at discharge? | Verdict |
|---|---|---|
| Demographics (age, gender, race) | yes | safe |
| Admission type/source, stay length, lab/procedure counts | yes — the stay is over | safe |
| **Prior** outpatient/emergency/inpatient visits | yes — history before this stay | safe |
| Prior diagnoses, medications, lab results taken during the stay | yes | safe |
| Engineered `prior_visits_total`, `n_meds_active`, `n_dose_changes` | yes — built from stay/history data | safe |
| Deceased rows (`discharge_disposition_id` 11/19/20/21) | — | **removed in notebook 01** (target leakage) |
| Row IDs (`encounter_id`, `patient_nbr`) | identifiers, not clinical facts | **removed from features** |

**What was deliberately NOT used:** nothing from after the discharge decision.
"""))

cells.append(nbf.v4.new_code_cell("""# Proof that no identifier remains in the feature set
leaky = [c for c in FEATURES if c in ("encounter_id", "patient_nbr")]
print("Identifier columns still in features:", leaky if leaky else "none")
print("Total features:", len(FEATURES))"""))

cells.append(nbf.v4.new_markdown_cell("""
## 6. Split by **patient**, not by row

23.4% of patients have more than one stay. If we split randomly by row, the same
person appears on both sides, stays in a family are near-identical, and the test
score is flattered.

`GroupShuffleSplit(groups=patients)` is the standard sklearn tool for this: it
puts whole groups (patients) on one side, never both.
"""))

cells.append(nbf.v4.new_code_cell("""splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(X, y, groups=patients))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
pat_train, pat_test = patients.iloc[train_idx], patients.iloc[test_idx]

print("train stays:", len(X_train), "| test stays:", len(X_test))
print("train patients:", pat_train.nunique(), "| test patients:", pat_test.nunique())
print()
print("positive rate train:", round(y_train.mean(), 4))
print("positive rate test :", round(y_test.mean(), 4))
print()
print("patients appearing on BOTH sides:",
      len(set(pat_train) & set(pat_test)), "<- must be zero")"""))

cells.append(nbf.v4.new_markdown_cell("""
**Sanity checks to read from the output above:**
- the overlap count must be **0** — that is the whole point of this split;
- the positive rates of train and test should be close (~0.11), proving the
  split did not accidentally create an easy or impossible test pile.
"""))

cells.append(nbf.v4.new_code_cell("""# --- Save the split for the later notebooks and the Streamlit app ---------------
X_train.to_csv(PROC / "X_train.csv", index=False)
X_test.to_csv(PROC / "X_test.csv", index=False)
y_train.to_csv(PROC / "y_train.csv", index=False)
y_test.to_csv(PROC / "y_test.csv", index=False)

with open(PROC / "feature_columns.json", "w") as f:
    json.dump(FEATURES, f, indent=1)

print("saved splits + feature list to", PROC)"""))

cells.append(nbf.v4.new_markdown_cell("""
## 7. The number to beat

Before any model, we write down what a **do-nothing model** scores on the test
set — it reflects the 11% class imbalance and every real model must clearly
exceed it.
"""))

cells.append(nbf.v4.new_code_cell("""# Always predict "not readmitted"
majority_rate = (y_test == 0).mean()
print(f"Test set majority-class rate (baseline 'accuracy'): {majority_rate:.2%}")
print("A real model must beat that while also catching actual readmissions,")
print("which is why we will read the confusion matrix, not accuracy.")"""))

nb["cells"] = cells
nbf.write(nb, NB_PATH)
print("appended; notebook now has", len(cells), "cells ->", NB_PATH)
