"""
BUILDER PART 4 -- appends diagnoses/ICD-9, correlations, Part-1 summary,
and saves the cleaned dataset for notebook 02.
"""
import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = PROJECT_ROOT / "notebooks" / "01_data_understanding.ipynb"

nb = nbf.read(NB_PATH, as_version=4)
cells = nb["cells"]

# ============================================================ 8. diagnoses
cells.append(nbf.v4.new_markdown_cell("""
## 8. Diagnosis codes (ICD-9) — turning ~17,000 codes into 9 clinical groups

`diag_1/2/3` are raw ICD-9 codes (`428`, `250.83`, `V58`, ...). Trees can split
on the raw string, but a **grouped** version is far more interpretable and
matches how clinicians talk (circulatory, respiratory, ...).

Grouping logic (a standard shortcut from the readmission literature):
- `250.xx`                       → diabetes
- 390–459 / 785                  → circulatory
- 460–519 / 786                  → respiratory
- 520–579 / 787                  → digestive
- 140–239                        → neoplasms
- 800–999                        → injury
- 710–739                        → musculoskeletal
- 580–629 / 788                  → genitourinary
- codes starting `V` or `E`      → supplementary (not a disease)
- anything else / missing        → other
"""))

cells.append(nbf.v4.new_code_cell("""def icd9_group(code):
    \"\"\"Map one ICD-9 code to a coarse clinical group.\"\"\"
    if pd.isna(code):
        return "missing"
    s = str(code).strip()
    if s[0] in ("V", "E"):
        return "supplementary"
    try:
        n = float(s.split(".")[0])
    except ValueError:
        return "other"
    if 250 <= n < 251:                     # diabetes
        return "diabetes"
    if (390 <= n < 460) or n == 785:
        return "circulatory"
    if (460 <= n < 520) or n == 786:
        return "respiratory"
    if (520 <= n < 580) or n == 787:
        return "digestive"
    if 140 <= n < 240:
        return "neoplasms"
    if 800 <= n < 1000:
        return "injury"
    if 710 <= n < 740:
        return "musculoskeletal"
    if (580 <= n < 630) or n == 788:
        return "genitourinary"
    return "other"

for c in ["diag_1", "diag_2", "diag_3"]:
    df[f"{c}_grp"] = df[c].map(icd9_group)

groups = pd.DataFrame({
    "diag_1": df["diag_1_grp"].value_counts(),
    "diag_2": df["diag_2_grp"].value_counts(),
    "diag_3": df["diag_3_grp"].value_counts(),
})
print(groups.to_string())"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 8: readmit rate per diagnosis group, by position -------------------
data = []
for c in ["diag_1", "diag_2", "diag_3"]:
    g = (df.groupby(f"{c}_grp", observed=True)["readmit_30d"]
           .agg(n="count", readmit_rate="mean").reset_index())
    g["column"] = c
    data.append(g)
icd = pd.concat(data)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=icd, x="diag_1_grp", y="readmit_rate", hue="column", ax=ax)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Figure 7 — Readmission rate by diagnosis group and position")
ax.set_xlabel("diagnosis group")
ax.set_ylabel("readmission rate")
ax.tick_params(axis="x", rotation=30)
ax.axhline(df["readmit_30d"].mean(), color="grey", ls="--", lw=1)
plt.tight_layout()
plt.show()

print("Readmit rate per group (diag_1):")
print(icd[icd["column"] == "diag_1"].set_index("diag_1_grp")
        .round(4)[["n", "readmit_rate"]].sort_values("readmit_rate", ascending=False).to_string())"""))

# ============================================================ 9. correlations
cells.append(nbf.v4.new_markdown_cell("""
## 9. Numeric correlations with the target

Correlation answers only *"does this number move together with readmission?"* —
it cannot capture non-linear effects (like prior visits, where risk jumps at 1
stay and keeps rising), which is why we use it for triage, not as proof.
"""))

cells.append(nbf.v4.new_code_cell("""corr = (df[num_cols + ["readmit_30d"]].corr()["readmit_30d"]
          .drop("readmit_30d").sort_values(key=abs, ascending=False).round(3))
print(corr.to_string())"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 9: correlation heatmap ---------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df[num_cols + ["readmit_30d"]].corr(), annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, square=True, ax=ax, annot_kws={"size": 8})
ax.set_title("Figure 8 — Correlation between numeric features and the target")
plt.tight_layout()
plt.show()"""))

# ============================================================ 10. summary
cells.append(nbf.v4.new_markdown_cell("""
## 10. Part 1 summary — what we learned about the data

**Structure**
- 101,766 stays of 71,518 diabetic patients; one row = one admission.
- 23.4% of patients have more than one stay (up to 40) — so we must split by
  **patient** (done in Part 2) or the test set leaks.

**Target**
- Binary `readmit_30d` = (`readmitted == "<30"`).
- **11.34% positives (11,357 of 100,114 rows)** — imbalanced, so accuracy is a trap.

**Data quality**
- `weight` missing for ~97% → will be dropped (executed in notebook 02).
- `medical_specialty`, `payer_code`, lab results missing for 40–95% — we keep them
  as their own "Missing" category because *being untested* is itself informative.
- Deceased patients (`discharge_disposition_id` 11/19/20/21) removed: they can
  never be readmitted → target leakage.
- Constant drug columns `examide` / `citoglipton` (single value `No`) dropped.

**Useful patterns**
1. `number_inpatient` is the standout: **8% → 45%** as prior stays go 0 → 8.
2. Discharge destination spreads from ~4.8% (hospice) to **36.7%** (psychiatric
   units); the 0% group (expired) was removed as leakage.
3. Utilisation counts (medications, diagnoses, labs) are all slightly higher for
   readmitted patients.
4. Age is non-linear: young adults (20–30) readmit most, seniors slightly less.
5. Race and gender are essentially flat — weak predictors and a fairness risk.
6. Diagnoses matter modestly; neoplasms as a secondary diagnosis stand out.
"""))

cells.append(nbf.v4.new_code_cell("""# --- Save the structurally-cleaned dataset for notebook 02 --------------------
OUT = PROJECT_ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# The raw 3-way target has served its purpose; the processed file carries the binary target
df_out = df.drop(columns=["readmitted"])
out_path = OUT / "diabetes_clean_stage1.csv"
df_out.to_csv(out_path, index=False)
print("Saved:", out_path)
print("Shape written:", df_out.shape)
print("Columns:", len(df_out.columns))"""))

cells.append(nbf.v4.new_markdown_cell("""
---
**End of Part 1.** Next notebook: `02_preprocessing_and_modeling.ipynb`
(missing-value encoding, patient-level train/test split, three models,
evaluation, SHAP, and the healthcare discussion).
"""))

nb["cells"] = cells
nbf.write(nb, NB_PATH)
print("appended; notebook now has", len(cells), "cells ->", NB_PATH)
