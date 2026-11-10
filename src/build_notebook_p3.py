"""
BUILDER PART 3 -- appends feature-pattern sections + charts to notebook 01.
"""
import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = PROJECT_ROOT / "notebooks" / "01_data_understanding.ipynb"

nb = nbf.read(NB_PATH, as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell("""
## 7. Which features might influence readmission? (Part 1, Q4)

Method: for every candidate feature, compute the readmission rate **within each
group** and look at the spread. A useful feature separates the groups; a flat
profile tells us the feature carries little signal on its own.

`groupby(...)["readmit_30d"].mean()` answers: *"among all stays with this
feature value, what fraction were readmitted within 30 days?"* — that fraction
is the readmission rate for the group.
"""))

cells.append(nbf.v4.new_code_cell("""def rate_table(col, min_n=100):
    \"\"\"Readmission rate per value of `col`, keeping only groups with n >= min_n.\"\"\"
    g = (df.groupby(col, observed=True)["readmit_30d"]
           .agg(n="count", readmit_rate="mean")
           .sort_values("readmit_rate", ascending=False))
    return g[g["n"] >= min_n].round(4)      # drop noise-prone tiny groups

for col in ["age", "race", "gender", "diabetesMed", "change", "insulin"]:
    print()
    print(f"--- {col} ---")
    print(rate_table(col).to_string())"""))

cells.append(nbf.v4.new_markdown_cell("""
### The ID columns need `IDS_mapping.csv` to be readable

`discharge_disposition_id = 6` is meaningless until we look up the mapping file,
which tells us 6 = "Discharged/transferred to home with home health service".

The file is really **three tables glued together** with blank separator rows:
`admission_type_id`, `discharge_disposition_id`, `admission_source_id`.
The helper below slices out one table by taking every row *after* its title
until the next title, and returns a plain `{id: description}` dictionary.
"""))

cells.append(nbf.v4.new_code_cell("""MAP = PROJECT_ROOT / "data" / "raw" / "IDS_mapping.csv"
raw = pd.read_csv(MAP, header=None, names=["id", "description"],
                  skip_blank_lines=True, dtype=str)

TITLES = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

def split_map(start_title):
    \"\"\"Return {id_number: description} for one of the 3 stacked tables.\"\"\"
    start = raw.index[raw["id"] == start_title][0]
    later = [i for i in raw.index[raw["id"].isin(TITLES)] if i > start]
    end = min(later) if later else len(raw)         # stop before the next table
    block = raw.iloc[start + 1:end]
    block = block[block["id"].notna() & (block["id"].str.strip() != "")]
    return {int(str(i).strip()): str(d).strip()
            for i, d in zip(block["id"], block["description"])}

disc_map = split_map("discharge_disposition_id")
adm_map  = split_map("admission_type_id")
src_map  = split_map("admission_source_id")

print("mapping sizes -> admission:", len(adm_map),
      "| discharge:", len(disc_map), "| source:", len(src_map))
print("discharge 1 =", disc_map.get(1))
print("discharge 3 =", disc_map.get(3))
print("discharge 11 =", disc_map.get(11))"""))

cells.append(nbf.v4.new_code_cell("""# Readmission rate by discharge destination, with human-readable labels
g = (df.groupby("discharge_disposition_id", observed=True)["readmit_30d"]
       .agg(n="count", readmit_rate="mean"))
g["destination"] = g.index.map(disc_map)            # dict lookup: id -> label
g = g[g["n"] >= 100].sort_values("readmit_rate", ascending=False)
print(g.round(4).to_string())

print()
print("NOTE: the lowest groups are hospice / long-term-care patients who have a")
print("very low risk by construction; the deceased rows were removed earlier.")"""))

cells.append(nbf.v4.new_markdown_cell("""
### Chart: readmission rate by discharge destination
"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 3: readmit rate by discharge destination ---------------------------
plot = g.head(12)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=plot["readmit_rate"], y=plot["destination"], ax=ax, palette="Reds_r")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Figure 3 - Readmission rate by discharge destination (n>=100)")
ax.set_xlabel("readmitted within 30 days")
for y, v in enumerate(plot["readmit_rate"]):
    ax.text(v + 0.002, y, f"{v:.1%}", va="center", fontsize=9)
ax.set_xlim(0, plot["readmit_rate"].max() * 1.18)
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""
### The strongest single pattern: prior inpatient visits
"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 4: the monotone relationship --------------------------------------
g2 = (df.groupby("number_inpatient", observed=True)["readmit_30d"]
        .agg(n="count", readmit_rate="mean"))
g2 = g2[g2["n"] >= 50]                      # skip tiny high-count groups

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(g2.index, g2["readmit_rate"], marker="o", color="#4C72B0")
ax.axhline(df["readmit_30d"].mean(), color="grey", ls="--", label="overall rate")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel("number of inpatient visits in the previous year")
ax.set_ylabel("readmission rate (<30 days)")
ax.set_title("Figure 4 - More prior stays = higher readmission risk")
ax.legend()
for x, y in zip(g2.index, g2["readmit_rate"]):
    ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=8)
ax.set_ylim(0, g2["readmit_rate"].max() * 1.25)
plt.tight_layout()
plt.show()

print(g2.round(4).to_string())"""))

cells.append(nbf.v4.new_markdown_cell("""
### Age, utilisation, and diagnosis codes
"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 5: age distribution + age vs readmit rate -------------------------
age_order = sorted(df["age"].dropna().unique())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(data=df, x="age", ax=ax[0], color="#55A868")
ax[0].set_title("Stays per age bracket")
ax[0].tick_params(axis="x", rotation=45)

by_age = (df.groupby("age", observed=True)["readmit_30d"].mean()
            .reindex(age_order))
ax[1].plot(age_order, by_age.values, marker="o", color="#DD8452")
ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax[1].set_title("Readmission rate by age")
ax[1].tick_params(axis="x", rotation=45)
plt.suptitle("Figure 5 - Most stays are older patients, but young adults readmit the most",
             y=1.03, fontsize=12)
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 6: utilisation distributions by outcome ---------------------------
num_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses"]

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, c in zip(axes.ravel(), num_cols):
    sns.histplot(data=df, x=c, hue="readmit_30d", ax=ax, bins=30,
                 palette={0: "#4C72B0", 1: "#DD8452"}, alpha=0.6)
    ax.set_title(c, fontsize=10)
plt.suptitle("Figure 6 - Numeric features by outcome (orange = readmitted <30d)",
             y=1.02, fontsize=12)
plt.tight_layout()
plt.show()"""))

nb["cells"] = cells
nbf.write(nb, NB_PATH)
print("appended; notebook now has", len(cells), "cells ->", NB_PATH)
