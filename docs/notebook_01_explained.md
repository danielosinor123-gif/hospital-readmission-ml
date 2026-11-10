# Notebook 01, explained for someone new to data science

This file explains, in plain language, what every part of
`notebooks/01_data_understanding.ipynb` is doing and why.

A Jupyter notebook is simply a document where you alternate:
- **markdown cells** (text, like this file), and
- **code cells** (Python you can run, whose printed results and charts appear
  right underneath).
It is used because analysis is a *story*: every step is explained, then shown.

---

## Chapter 0 — "Borrowing other people's tools" (the imports)

Python itself is a basic language. Almost everything useful lives in
**libraries** = free toolboxes other people wrote. `import` means "open that
toolbox and let me use it".

| Import | Plain meaning | What we use it for |
|---|---|---|
| `pandas as pd` | "spreadsheet in Python" | Load the CSV file into a table and slice/group it |
| `numpy as np` | "fast maths" | Anything numeric (arrays, sorting, maths on columns) |
| `matplotlib.pyplot as plt` | "the drawing engine" | Every chart in the notebook is ultimately drawn by matplotlib |
| `seaborn as sns` | "matplotlib with nicer defaults" | One-line statistical charts (bar charts, histograms) |
| `matplotlib.ticker as mtick` | "axis number formatting" | Turns 0.1134 into "11.3%" on chart axes |
| `pathlib.Path` | "smart file addresses" | Builds paths that work on any computer, so the notebook is not tied to one machine |
| `warnings` | "silence the noise" | Hides harmless complaints so the output stays readable |

Two more notebook-only details:
- `sns.set_theme(...)` = pick the visual style of all charts up front.
- `plt.rcParams["figure.figsize"] = (9, 5)` = make charts a comfortable size.

---

## Chapter 1 — Opening the data

```python
df = pd.read_csv(CSV, na_values=["?"], low_memory=False)
```

`df` is short for **DataFrame** — the pandas word for "a table with rows and
named columns".

Two options matter:
- **`na_values=["?"]`** — this file marks "no data recorded" with a question
  mark. If we do not say so, Python thinks `"?"` is a real answer and we would
  never notice the holes.
- **`low_memory=False`** — one column (`payer_code`) mixes text and numbers, so
  pandas warns about guessing its type. This stops the warning.

Result printed: **101,766 rows, 50 columns** — but only **71,518 patients**.
One row = **one hospital stay**, not one person. That distinction causes the
single biggest trap in this project (see Chapter 5).

---

## Chapter 2 — Cleaning that stops the model cheating

### 2.1 Removing patients who died (the important one)

`discharge_disposition_id` tells us where the patient went after the stay.
Some codes mean the patient **died** (11 = "Expired", 19/20/21 = died in hospice).

Look at the printed proof: all 1,652 such rows have `readmitted = NO` — because
a dead person cannot return.

If we kept them, the model would learn the shortcut
**"dead ⟹ never readmitted"**, which is 100% correct and 100% useless in the
real world. This kind of shortcut is called **data leakage**: information that
would not exist when we make a real prediction. Removing those rows is a
documented, defensible decision — and a reviewer will ask about it.

### 2.3 Dropping empty columns

`examide` and `citoglipton` contain the value `No` in every single row.
A column with only one value tells the model nothing (zero information), so it
goes. The code finds them automatically with
`nunique(dropna=False) <= 1` — "how many different values does this column
have? One or fewer means drop it."

---

## Chapter 3 — The question we actually want answered (the target)

The raw column `readmitted` has **three** outcomes:

| raw value | meaning |
|---|---|
| `NO` | not readmitted within the recorded period |
| `>30` | came back, but after 30 days |
| `<30` | came back within 30 days |

The hospital's question is *"who is coming back within 30 days?"* — that is a
yes/no question. So we build the target:

```python
df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)
```

Read it as: *"if this stay's readmitted value equals `<30`, set 1, otherwise 0."*
The `.astype(int)` turns True/False into 1/0.

**Why merge `>30` with `NO`?** Because the action the hospital takes (booking a
follow-up, arranging home care) is tied to the 30-day window. A model that
distinguishes `>30` from `NO` answers a different, easier question.

The chart on the right half shows the result: only **11,357 of 100,114 stays**
are a "1" — that imbalance drives everything later.

---

## Chapter 4 — The holes in the data (missing values)

Real hospital data is full of gaps. The notebook counts the gaps per column:

| column | % missing | what we do |
|---|---|---|
| `weight` | 97% | drop it — nothing survives |
| `max_glu_serum` | 95% | keep, label as `"Not measured"` |
| `A1Cresult` | 83% | keep, label as `"Not measured"` |
| `medical_specialty` | 49% | keep, label as `"Missing"` |
| `payer_code` | 40% | keep, label as `"Missing"` |
| `race` | 2.2% | keep, label as `"Missing"` |

The key idea: **a missing value is not always a problem to delete — sometimes it
is a clue.** For `A1Cresult`, the fact that a test was *not* done tells us about
how intensively that patient was managed. So we convert "no value" into its own
category the model can use, instead of throwing the column away.

---

## Chapter 5 — The same patient appearing many times

**17,773 of 71,518 patients (23.4%) have more than one stay** — one patient
appears 40 times.

This is the dangerous duplicate: it is not an exact copy of a row, it is the
*same person* in several stays. If we randomly split the rows into
"practice data" and "exam data", the same person would be in both — and since
patients are similar to themselves, the model would score far better on the exam
than it deserves.

The fix (done in notebook 02) is to split **by patient**, so all stays of one
person are on the same side.

---

## Chapter 6 — Why "accuracy" is a trap here

Only **11.34%** of stays are positives: **1 readmission for every 8.8 stays with
no readmission**.

Imagine a "model" that simply answers *"nobody gets readmitted"* to every
patient. It would be **right 88.66% of the time** — a great-looking number that
catches **zero** of the patients at risk.

So we cannot judge this project by accuracy. We need metrics that measure
"of the people we flagged, how many were right" (**precision**) and "of the
people who did come back, how many did we catch" (**recall**). That is Part 3.

---

## Chapter 7 — Which features matter? (the pattern hunt)

The technique, in plain words:

1. Take a column, e.g. `age`.
2. Split the table into groups by its values (`[70-80)`, `[80-90)`, ...).
3. In each group, ask: *what fraction were readmitted within 30 days?*

In pandas that is one line:

```python
df.groupby("age")["readmit_30d"].mean()
```

Translation: *"group the table by age, then for each group give me the average of
readmit_30d."* Since the target is 1 for "readmitted", an average of 0.12 means
"12% of those stays were readmissions."

We also always keep `n` (the group size) next to the rate — a rate computed from
3 patients is noise, not knowledge. That is why the helper keeps only groups with
at least 100 stays.

### What the notebook found (the findings you must be able to defend)

| finding | evidence | meaning |
|---|---|---|
| Prior inpatient visits dominate | 0 visits → 8.6%, 1 → 13.2%, 5 → 31.9%, 8 → 45.9% | The single strongest signal; risk rises smoothly, which makes it trustworthy |
| Discharge destination matters | psychiatric ward 36.7%, rehab 27.7%, SNF 14.7%, home 9.3%, hospice 4.8% | Where a patient goes after discharge encodes how fragile they are |
| Age is non-linear | children 1.9%, young adults 14.3%, seniors ~12% | Young diabetic adults are a hidden risk group |
| Utilisation counts | meds, labs, diagnoses slightly higher for readmitted | Sicklier stays cost more services |
| Race and gender | every group ≈ 11% | **Flat** — weak predictors, and a fairness risk if relied on |

### Decoding the ID columns

`discharge_disposition_id = 28` means nothing on its own. `IDS_mapping.csv` is
the hospital's decoder ring: 28 = "psychiatric hospital". The notebook has a
small function `split_map()` that reads that decoder (which is three tables glued
together) into a simple `{number: label}` dictionary — so the charts can show
readable words instead of mystery numbers.

---

## Chapter 8 — Turning diagnosis codes into something usable

`diag_1/2/3` are **ICD-9** codes — the international hospital billing codes for
diseases: `428` = heart failure, `250.83` = diabetes, and there are ~17,000 of
them in the file.

Two problems with raw codes:
1. A model can memorize rare codes and overfit.
2. A human cannot understand a split on `diag_1 = 414.01`.

So `icd9_group()` maps each code to one of ten plain clinical groups
(diabetes, circulatory, respiratory, ...) using well-established code ranges.
It is a standard industry shortcut, and it makes the charts and the later SHAP
explanations readable by a non-technical audience.

---

## Chapter 9 — Correlation (used carefully)

`df.corr()` measures, for each pair of numeric columns, whether they move
together, from −1 (opposite) to +1 (in sync).

Result: `number_inpatient` correlates **0.168** with the target — the highest,
but far from 1.0. That is normal and healthy: it means no single number fully
determines readmission, so a combination of features is genuinely needed.

**The catch the notebook states out loud:** correlation only detects
"straight-line" relationships. `number_inpatient` does not act in a straight line
(it jumps at the first prior visit and keeps climbing), so correlation
*understates* it. Use correlation to shortlist features, never as proof.

---

## Chapter 10 — Saving the result

The last cell writes the cleaned table to
`data/processed/diabetes_clean_stage1.csv`. This is the standard "raw data is
untouchable, cleaned data is what everything downstream uses" convention:

```
data/raw/        downloaded, never edited  ← the original evidence
data/processed/  what this notebook produces
```

If notebook 02 needs the data, it reads the processed file — not the raw one.

---

## Glossary (words your reviewer may use)

| term | plain meaning |
|---|---|
| DataFrame (`df`) | a table: rows + named columns |
| label / target | the column we are trying to predict (here: 1 = readmitted <30d) |
| feature | any input column the model learns from |
| class imbalance | one outcome is much rarer than the other (here 11% vs 89%) |
| data leakage | information that would not exist at prediction time sneaking into training |
| one-hot encoding | turning a category like "Caucasian" into 1/0 columns |
| overfitting | memorising the practice set instead of learning the rule |
| precision | of the patients we flagged, how many really came back |
| recall | of the patients who came back, how many we flagged |
| SHAP | a method that explains *which features pushed a single prediction* |
