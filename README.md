# Predicting 30-Day Hospital Readmission with Machine Learning

A complete, documented machine-learning pipeline built on the
**Diabetes 130-US Hospitals (1999–2008)** dataset from the UCI Machine Learning
Repository.

> **Status:** student / portfolio project. It demonstrates an honest end-to-end ML
> workflow. It is **not** a clinical device and must not be used for real medical
> decisions. See `report.md` § "Limitations".

---

## What this project does

Hospitals want to know, **at discharge**, which diabetic patients are likely to be
readmitted within 30 days, so that follow-up care can be arranged while the
patient is still in the building.

This repository covers the whole path: understanding the data → cleaning →
patient-safe splitting → three models → explainability → a working web app.

## Results at a glance

| Metric | Value | Plain meaning |
|---|---|---|
| Test ROC-AUC | see `report.md` | how well the model *ranks* patients by risk |
| Test PR-AUC | see `report.md` | the honest ranking score under an 11% positive rate |
| Test recall | see `report.md` | of the patients who really returned, the share we flag |
| Test precision | see `report.md` | of the patients we flag, the share who really return |

(Full tables and charts live in the notebooks; the headline numbers are in
`report.md`, which is generated from the same runs.)

---

## Repository layout

```
hospital_readmission_ml/
├── data/
│   ├── raw/                          # downloaded original data (never edited)
│   │   ├── diabetic_data.csv
│   │   └── IDS_mapping.csv
│   └── processed/                    # outputs of the pipeline
│       ├── diabetes_clean_stage1.csv # notebook 01 output
│       ├── X_train.csv / X_test.csv  # the patient-level split
│       ├── y_train.csv / y_test.csv
│       ├── feature_columns.json      # the exact feature list the model expects
│       ├── model_metrics.json        # metrics from notebook 02
│       └── final_model_summary.json  # metrics from notebook 03
├── notebooks/
│   ├── 01_data_understanding.ipynb   # Part 1: what the data is, its problems, patterns
│   ├── 02_preprocessing_and_modeling.ipynb  # Part 2 + Part 3: split, 3 models, evaluation
│   └── 03_tuning_threshold_and_explainability.ipynb  # tuning, threshold, SHAP, fairness
├── src/                              # small scripts that build/run/verify the above
│   ├── download_data.py              # downloads the dataset (reproducible)
│   ├── run_notebook.py               # executes a notebook headlessly, reports failures
│   ├── check_notebook.py             # verifies a notebook: cells, figures, errors
│   ├── test_app.py                   # drives the Streamlit app without a browser
│   ├── 01_patterns.py / 02_groupby_demo.py
│   ├── build_notebook_p*.py          # the scripts that generated the notebooks
│   ├── build_nb02_p*.py / build_nb03_p*.py
│   └── diagnostics/                  # measurements taken while debugging
│       ├── timing_test.py            # how long one LightGBM fit really takes
│       ├── timing_test_small.py
│       └── debug_threshold.py        # reproduced the threshold-selection bug
├── app/
│   └── app.py                        # Part 5: Streamlit app
├── models/                           # trained models (joblib), so the app runs as cloned
├── docs/
│   └── notebook_01_explained.md      # plain-English walkthrough of notebook 01
├── report.md                         # Part 6: approach, results, challenges, recommendations
└── requirements.txt
```

> **Notes on what is committed:** the raw data CSVs (~19 MB) are committed
> **deliberately**, so a reviewer can run the notebooks immediately after cloning
> without re-downloading (`src/download_data.py` recreates them from UCI anyway).
> The **trained models are not committed** — the Random Forest artifact alone is
> 727 MB, past GitHub's file limit, and models are fully recreated by notebooks 02
> and 03. Run notebook 02 once (~15 minutes) to produce `models/lightgbm.joblib`
> for the Streamlit app.

---

## How to run everything

### 1. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.13, pandas 3.x, scikit-learn 1.9, LightGBM 4.7.

### 2. Get the data

```bash
python src/download_data.py        # downloads + unzips into data/raw/
```

### 3. Run the notebooks, in order

```bash
python src/run_notebook.py notebooks/01_data_understanding.ipynb
python src/run_notebook.py notebooks/02_preprocessing_and_modeling.ipynb
python src/run_notebook.py notebooks/03_tuning_threshold_and_explainability.ipynb
```

Or open them interactively:

```bash
python -m notebook            # then open notebooks/01_...ipynb in the browser
```

> **Windows note:** on this machine the `jupyter` command is a broken stub — use
> `python -m notebook` / `python -m nbconvert` instead. Pre-executed HTML versions
> of the notebooks are also exported next to them (`*.html`).

### 4. Run the web app

```bash
python -m streamlit run app/app.py
```

The app asks for patient details known at discharge, shows the predicted
probability of readmission within 30 days, a risk band, and validates every input.

### 5. Verify the notebooks

```bash
python src/check_notebook.py    # prints cells, figures and error count
python src/test_app.py          # drives the Streamlit app's code without a browser
```

---

## Key decisions (the short version)

| Decision | Why |
|---|---|
| Binary target `<30` days | matches the hospital's actual question and the 30-day action window |
| Deceased stays removed | a dead patient cannot be readmitted — the model would otherwise learn a shortcut |
| **Split by patient, not by row** | 23% of patients have several stays; a row-split leaks the same person into train and test |
| `class_weight` / `scale_pos_weight` | only 11% of stays are positives; the model would otherwise learn "say no to everyone" |
| Recall-weighted threshold | for triage, missing a real readmission is usually worse than a false alarm |
| SHAP explanations | a model the hospital cannot question is a model it will not use |
| Fairness breakdown by race/gender | a model can look good overall and still fail one group |

---

## Known environment quirks (documented, not hidden)

- **LightGBM crashes inside scikit-learn's parallel workers** on this Windows
  build ("access violation"), so cross-validation runs with `n_jobs=1`. Slower, correct.
- **One LightGBM fit takes ~2.5 minutes** on this machine, so hyperparameter
  search runs on a stratified 20,000-row subsample and the winner is re-trained
  once on the full training set (the standard coarse-to-fine trade-off).
- **`jupyter` on PATH is a broken stub**; use `python -m notebook` and
  `python -m nbconvert`.

---

## References

- Strack et al. (2014), *Impact of HbA1c Measurement on Hospital Readmission
  Rates*, BioMed Research International. https://doi.org/10.1155/2014/781670
- UCI dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999+2008

## AI-assistance disclosure

Parts of this project (code scaffolding, notebook text, this README) were drafted
with AI assistance. Every line was reviewed and understood by the author, who can
explain any part of it; all numbers were produced by running the code in this
repository, and all failures encountered were documented rather than hidden.
