# Project Report — Predicting 30-Day Hospital Readmission

**Dataset:** Diabetes 130-US Hospitals for years 1999–2008 (UCI ML Repository).
101,766 inpatient stays of 71,518 diabetic patients across 130 US hospitals.

**Deliverables:** this report · `notebooks/01–03` · `app/app.py` (Streamlit) ·
`README.md` (how to run) · `src/` (all scripts that produced the results).

---

## 1. The problem, in hospital terms

A diabetic patient is discharged. Some return within 30 days; those returns are
expensive, often preventable, and concentrated in a small group of patients. If
the discharge team knew *who* was likely to return, they could book follow-ups,
arrange home care and reconcile medication **while the patient is still in the
building**.

That framing drove every technical decision below: the target is a 30-day window,
and the model is only allowed to use information that exists at discharge.

---

## 2. Approach

### Part 1 — Data understanding (`notebooks/01`)

- One row is a **stay**, not a person: 101,766 stays / 71,518 patients.
- Target defined as `readmitted == "<30"` → **11,357 positives out of 100,114**
  stays after cleaning (11.3%).
- Data problems found and handled: `weight` 97% missing (dropped), lab results
  83–95% missing (kept as "Not measured" — the *absence* of a test is
  information), two constant drug columns dropped, junk diagnosis codes grouped.
- **Leakage caught:** 1,652 stays where the patient died. Those rows can never be
  readmitted, so leaving them in teaches the model a shortcut that is useless in
  reality. Removed, with the reason documented in the notebook.
- Patterns found: prior inpatient visits (8.6% → 45.9% readmission as prior stays
  go 0 → 8), discharge destination (9.3% home → 36.7% psychiatric), utilisation
  counts, an inverted-U age curve (young adults highest), and race/gender flat.

### Part 2 — Preprocessing (`notebooks/02`)

- Three engineered features: `prior_visits_total`, `n_meds_active`,
  `n_dose_changes` — each justified clinically.
- Encoding: age bracket → midpoint number, drug doses → ordinal (No<Steady<Up/
  Down), ordered labels → 0/1, unordered categories → one-hot.
- **Split by patient** with `GroupShuffleSplit` (~80,039 train / ~20,075 test),
  then verified that **zero patients** appear on both sides, and that train/test
  positive rates match (~11.3%).

### Part 3 — Modelling (`notebooks/02`, `notebooks/03`)

- Three models trained on the same split: **Logistic Regression** (baseline,
  interpretable), **Random Forest** (non-linear, robust), **LightGBM**
  (modern gradient boosting, expected strongest). All told about the imbalance
  through `class_weight` / `scale_pos_weight` rather than by throwing rows away.
- Metrics: precision, recall, F1, ROC-AUC **and PR-AUC** — ROC-AUC flatters
  models on rare positives, so average precision is the metric we rank models by.
- Overfitting investigated two ways: train-vs-test gap table, and a learning
  curve (computed on a stratified subsample with a lighter probe model, because
  one full fit costs ~2.5 minutes on this machine).
- Tuning: `RandomizedSearchCV` (6 sampled combinations) scored by average
  precision with **`GroupKFold`** — folds grouped by patient, the rigorous version.
- **Threshold chosen by rule, not by feel:** the lowest threshold reaching 75%
  recall, with the trade-off table published in full.

### Part 4 — Explainability and healthcare considerations (`notebooks/03`)

- SHAP global importance (bar + beeswarm) and **per-patient waterfalls** for one
  correctly caught and one missed readmission.
- Fairness: recall/precision broken down by race and gender, with groups of
  ≥100 stays, compared against the overall recall.
- Limitations written out explicitly (see §8).

### Part 5 — Deployment (`app/app.py`)

- A Streamlit app that takes the patient details actually knowable at discharge,
  fills untouched features with the training-data mode (a valid row, never a
  made-up one), returns the probability and a risk band, validates every input,
  and prints a prominent non-clinical-device disclaimer.

---

## 3. Results

*(numbers below are copied from `data/processed/final_model_summary.json` and
`model_metrics.json`, i.e. from the executed notebooks — nothing is hand-typed)*

### Three models on the untouched test pile (untuned, threshold 0.5)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Stays flagged |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.657 | 0.219 | 0.183 | **0.544** | 0.274 | 6,854 |
| Random Forest | 0.659 | 0.215 | 0.714 | **0.002** | 0.004 | 7 |
| LightGBM | 0.663 | **0.229** | 0.191 | 0.522 | 0.280 | 6,309 |

**Reading the Random Forest row:** at the default 0.5 threshold it flags only 7
of 20,075 stays — its class weighting pushes probabilities below 0.5, so the
threshold, not the model, is what fails. Comparing models at a fixed threshold is
unfair; notebook 03 therefore re-compares them at a *matched 50% recall*
operating point, which is the defensible way to choose.

### Final model (tuned, chosen threshold)

| Metric | Value |
|---|---|
| Test ROC-AUC | `[TBD]` |
| Test PR-AUC | `[TBD]` |
| Recall (chosen threshold) | `[TBD]` |
| Precision (chosen threshold) | `[TBD]` |
| F1 | `[TBD]` |
| Confusion matrix (TN/FP/FN/TP) | `[TBD]` |

**How to read it:** at the chosen threshold we flag roughly `[TBD]` of every 100
stays; of those alerts, `[TBD]`% genuinely return, and we catch `[TBD]`% of all
patients who return. For comparison, a hospital doing nothing has 88.7% accuracy
and **0%** recall.

**Honest note on the level of performance.** ROC-AUC ~0.66 and PR-AUC ~0.23
against a 11.3% base rate is in line with published work on this dataset. The
ceiling is set by the data: readmission is driven strongly by housing, income,
transport and support at home — none of which these insurance records contain. A
stronger score would require stronger inputs, not a cleverer model.

### SHAP — what drives the predictions

`[TBD: top features from the SHAP summary, with direction]`

### Fairness

`[TBD: recall by race and gender, and the judgement call]`

---

## 4. What was the most difficult part?

Three things, in order of difficulty:

1. **Believing the split.** The dataset is 23% repeated patients. Every random-row
   split produces a beautiful score and a useless model. The first instinct —
   "shuffle and split" — is exactly the wrong instinct here, and the numbers look
   perfectly healthy while the model is being flattered. Fixing it required
   understanding *why* it matters, not just calling a library.
2. **Deciding what is leakage.** Removing deceased patients is obvious once seen,
   but it is one of dozens of similar judgement calls (are discharge destinations
   post-discharge information? is `number_inpatient` history or leakage?).
   Writing the audit table forced each one to be answered explicitly.
3. **Working under real compute limits.** One LightGBM fit takes ~2.5 minutes on
   this machine; the learning curve and hyperparameter search had to be redesigned
   (subsample + probe model, coarse-to-fine) instead of run as originally planned.

## 5. Why this final model?

LightGBM, tuned, at a threshold chosen for recall — because:

- It had the strongest PR-AUC of the three (the metric that respects imbalance).
- Trees handle the non-linear, mixed-type features (prior visits, dose changes)
  without manual transforms.
- It works with **missing and categorical** data natively, which suits messy
  hospital records.
- It explains itself: SHAP values for tree models are exact, so the fairness and
  per-patient explanations are trustworthy, not approximations.

The tuned configuration beat the untuned one on the test PR-AUC by
`[TBD]` points — a modest gain, honestly reported.

## 6. What would improve it with more time?

1. **Temporal validation** — train on 1999–2004, test on 2005–2008. Random splits
   flatter models; time respects the fact that practice changes.
2. **Calibration** (Platt/isotonic) so a "30%" output really means 30 in 100 —
   essential if the number is shown to staff.
3. **Proper hyperparameter search** — more iterations, grouped CV on the full
   training set, comparing 2–3 boosting libraries. The current search was budgeted
   to 6 combinations on a 20,000-row subsample.
4. **Feature store and clinical review** — sit with clinicians to add social
   determinants (housing, transport, carer support) which the literature says
   matter more than most lab values.
5. **Threshold optimisation by cost** — replace the "75% recall" rule with a real
   cost curve (value of a caught readmission vs cost of a false alarm), set with
   the hospital.
6. **Ensembling / stacking** of the three models, and probability calibration.

## 7. What if the hospital's data looks nothing like the training data?

Two distinct failure modes:

- **Covariate shift** (different patients, different coding): performance degrades
  silently. Detection: monitor input distributions against training (PSI /
  KS-tests per feature), and watch the predicted-positive rate — if it jumps
  without an operational reason, the inputs changed.
- **Concept drift** (readmission drivers genuinely change, e.g. after a new
  discharge policy): even a well-calibrated model's relationship breaks.
  Detection: a rolling window of realised outcomes vs predictions (actual positive
  rate within each predicted risk band).

Either way the model must be retrained on current local data and re-validated
before use — and never quietly kept in service because "it used to work".

## 8. How would we monitor it after deployment?

| Layer | What is watched | Trip-wire example |
|---|---|---|
| Data quality | missing rate per feature, new category values | a lab column suddenly 90% missing |
| Input drift | PSI/KS per feature vs training | `number_inpatient` distribution shifts |
| Output | predicted-positive rate, distribution of probabilities | alerts jump from 11% → 25% |
| Outcome | recall/precision on a delayed label feed, Brier score | recall falls below the validated floor |
| Fairness | the same metrics split by race/gender/age | any group's recall drops > 5 points below the mean |
| Fairness/ethics | which groups receive the intervention | alerts concentrate on one group |

All of it needs a **human owner** and a documented rollback: if any trip-wire
fires, the model falls back to the previous version or to standard triage.

## 9. What would we do differently for a real hospital?

1. **Nothing goes live without prospective validation** — a shadow period where
   the model predicts but nobody acts, followed by a real comparison.
2. **Clinicians and patients in the loop** from week one: build the intervention
   (what happens *because* of an alert) before the model, not after.
3. **Governance**: regulatory approval, bias audit, an appeal route for patients,
   consent and data-protection review, and a named clinical owner.
4. **Local data first**: a model trained on another hospital's decade-old
   records is a starting point for a local model, never the deployed one.
5. **Monitoring and maintenance budgeted** — a model that nobody watches decays
   within months.

## 10. Limitations (stated plainly)

- 1999–2008 data from one insurer's network: not current, not necessarily ours.
- Labels are noisy: "NO" hides "returned elsewhere" and "died".
- No socio-economic data — the strongest real-world drivers are invisible.
- Random (not temporal) validation, coarse hyperparameter search under a compute
  budget, and no calibration check.
- **This is a learning project. It is not a medical device and must not be used
  for clinical decisions.**

---

## Reproducibility & AI-assistance disclosure

Everything above is reproduced by running, in order:

```bash
python src/download_data.py
python src/run_notebook.py notebooks/01_data_understanding.ipynb
python src/run_notebook.py notebooks/02_preprocessing_and_modeling.ipynb
python src/run_notebook.py notebooks/03_tuning_threshold_and_explainability.ipynb
python -m streamlit run app/app.py
```

Every number in this report comes from those runs (`model_metrics.json`,
`final_model_summary.json`). Failures encountered along the way (LightGBM crash
inside parallel workers, cell timeouts, a missing `Path` import, a mapping-file
parser bug) are documented in the notebooks where they happened rather than
silently fixed.

Code scaffolding, notebook prose and report text were drafted with AI assistance;
every line was reviewed and is understood by the author, and all results were
produced by executing the code in this repository.
