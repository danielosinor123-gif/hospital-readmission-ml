"""
BUILDER PART 2 -- appends to notebooks/01_data_understanding.ipynb:
target definition + class balance + missing values + duplicates/inconsistencies.
"""
import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = PROJECT_ROOT / "notebooks" / "01_data_understanding.ipynb"

nb = nbf.read(NB_PATH, as_version=4)
cells = nb["cells"]

# ============================================================ 3. target
cells.append(nbf.v4.new_markdown_cell("""
## 3. The target variable (Part 1, Q2)

`readmitted` has **three** values, but the hospital's question is binary:
*"will this patient come back within 30 days?"*

| raw value | meaning | our label |
|---|---|---|
| `<30` | readmitted within 30 days | **1** |
| `>30` | readmitted, but later than 30 days | 0 |
| `NO` | not readmitted in the recorded window | 0 |

> Why merge `>30` with `NO`? Because the operational decision (book a follow-up,
> arrange home care) only exists for the 30-day window. A model that separates
> `>30` from `NO` answers a different, easier question than the one asked.
"""))

cells.append(nbf.v4.new_code_cell("""print("Raw 'readmitted' values:")
print(df["readmitted"].value_counts().to_string())

# Binary target: 1 = came back within 30 days
df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)

print()
print("Binary target counts (0 = not <30 days, 1 = readmitted <30 days):")
print(df["readmit_30d"].value_counts().to_string())
print()
print(f"Positive rate: {df['readmit_30d'].mean():.4f}  (~1 positive per "
      f"{1/df['readmit_30d'].mean():.1f} negatives)")"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 1: the three raw categories vs the binary target -------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

counts = df["readmitted"].value_counts().reindex(["<30", ">30", "NO"])
sns.barplot(x=counts.index, y=counts.values, ax=ax[0], palette="Blues_r")
ax[0].set_title("Raw outcome per stay")
ax[0].set_xlabel("readmitted")
ax[0].set_ylabel("number of stays")
for i, v in enumerate(counts.values):
    ax[0].text(i, v + 800, f"{v:,}\\n({v/len(df):.1%})", ha="center", fontsize=9)

pos = int(df["readmit_30d"].sum())
neg = int((df["readmit_30d"] == 0).sum())
ax[1].bar(["0  not <30d", "1  <30d"], [neg, pos], color=["#4C72B0", "#DD8452"])
ax[1].set_title("Binary training target (imbalanced)")
ax[1].set_ylabel("number of stays")
for i, v in enumerate([neg, pos]):
    ax[1].text(i, v + 800, f"{v:,}\\n({v/len(df):.1%})", ha="center", fontsize=9)

plt.suptitle("Figure 1 - The readmission target", y=1.02, fontsize=13)
plt.tight_layout()
plt.show()

print("Accuracy of a model that ALWAYS predicts 0: "
      f"{(df['readmit_30d'] == 0).mean():.2%}  <-- trivial and misleading")"""))

# ============================================================ 4. missing
cells.append(nbf.v4.new_markdown_cell("""
## 4. Missing values (Part 1, Q3 part B)

Missingness is reported **per column**, sorted, so we can separate
"small gap" (fix it) from "mostly absent" (drop the column or treat as signal).
"""))

cells.append(nbf.v4.new_code_cell("""missing = pd.DataFrame({
    "n_missing": df.isna().sum(),
    "pct_missing": (df.isna().mean() * 100).round(2),
})
missing = missing[missing["n_missing"] > 0].sort_values("pct_missing", ascending=False)
print(missing.to_string())
print()
print(f"Columns with no missing values: {df.shape[1] - len(missing)} of {df.shape[1]}")"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 2: missing-data bar chart ------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
missing_sorted = missing.sort_values("pct_missing")
ax.barh(missing_sorted.index, missing_sorted["pct_missing"], color="#C44E52")
ax.set_xlabel("% of rows missing")
ax.set_title("Figure 2 - Missing values per column")
for y, (idx, row) in enumerate(missing_sorted.iterrows()):
    ax.text(row["pct_missing"] + 1, y, f"{row['pct_missing']:.0f}%", va="center", fontsize=9)
ax.set_xlim(0, 108)
plt.tight_layout()
plt.show()

print("Reading: 'weight' is 97% missing -> unusable.")
print("'medical_specialty' ~49% missing -> the absence itself may be informative,")
print("so we keep it as its own category instead of dropping it.")"""))

cells.append(nbf.v4.new_markdown_cell("""
### Decision table for missingness (each decision is justified)

| Column | % missing | Decision | Why |
|---|---|---|---|
| `weight` | ~97% | **Drop column** | nothing left to learn from |
| `medical_specialty` | ~49% | Keep, fill `"Missing"` | "no specialist" is a real signal |
| `payer_code` | ~40% | Fill `"Missing"` | low value, kept for completeness |
| `max_glu_serum` | ~95% | Fill `"Not measured"` | *whether* a test ran is information |
| `A1Cresult` | ~83% | Fill `"Not measured"` | same |
| `race` | ~2% | Fill `"Missing"` | tiny share, negligible distortion |
| `diag_1/2/3` | up to ~1.5% | Fill `"Missing"` | handled by ICD-9 grouping in section 8 |
"""))

cells.append(nbf.v4.new_markdown_cell("""
## 5. Duplicates and inconsistencies (Part 1, Q3 part C)

- **Exact duplicate rows?** The dataset contains nearly none, but we always
  check — duplicates would let the same information sit in both train and test.
- **Repeated patients?** Yes, and that is the dangerous kind: the same
  `patient_nbr` appears in many stays (16,483 patients *after* the deceased rows
  were removed in section 2 — section 1 counted 16,773 *before* that removal).
  Handled by a **patient-level split** in notebook 02.
- **Inconsistent categories?** `gender` has a few `Unknown/Invalid` rows; several
  ID codes mean "Not mapped"/"Unknown"; `diag` fields contain junk values; some
  drug columns are constant.
"""))

cells.append(nbf.v4.new_code_cell("""print("Exact duplicate rows:", df.duplicated().sum())

per_patient = df["patient_nbr"].value_counts()
print("Patients with more than one stay:", (per_patient > 1).sum(),
      f"({(per_patient > 1).mean():.1%} of patients)")
print("Most stays by a single patient:", per_patient.max())

print()
print("gender:", df["gender"].value_counts(dropna=False).to_dict())

# Drug columns should only contain No / Steady / Up / Down
drug_cols = [c for c in df.columns
             if set(df[c].dropna().unique()) <= {"No", "Steady", "Up", "Down"}]
print("Drug-dose columns (No/Steady/Up/Down only):", len(drug_cols))
print("Example:", drug_cols[:4], "...")"""))

# ============================================================ 6. balance
cells.append(nbf.v4.new_markdown_cell("""
## 6. Is the dataset balanced? (Part 1, Q5)

**No — it is imbalanced: only ~11% of stays are positives (~1 : 8).**

Consequences we handle in Parts 2-3:
- Accuracy is a misleading headline metric (always predicting "no" scores ~89%).
- Use **recall / precision / F1 / PR-AUC**, plus `class_weight="balanced"` or
  resampling so the model does not simply learn to say "no" to everyone.
"""))

cells.append(nbf.v4.new_code_cell("""pos_rate = df["readmit_30d"].mean()
print(f"Positives: {df['readmit_30d'].sum():,}  ({pos_rate:.2%})")
print(f"Negatives: {(df['readmit_30d'] == 0).sum():,}  ({1-pos_rate:.2%})")
print(f"Imbalance ratio: 1 positive : {1/pos_rate:.1f} negatives")"""))

nb["cells"] = cells
nbf.write(nb, NB_PATH)
print("appended; notebook now has", len(cells), "cells ->", NB_PATH)
