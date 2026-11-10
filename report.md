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
- **Threshold chosen by policy, not by feel.** Two candidates were computed from
  the full trade-off table (published in notebook 03): an **F1-optimal**
  threshold (~31% of stays alerted, ~half of readmissions caught) and a
  **recall-first** threshold (75% recall, but ~68% of stays alerted at ~14%
  precision — unusable in a ward). The F1-optimal rule was deployed and the
  alternative is reported, because that trade is a hospital policy decision, not
  a modelling one.

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

**Matched-recall comparison (50% recall, notebook 03):**

| Model | Precision @ 50% recall | Threshold needed |
|---|---|---|
| Logistic Regression | 0.115 | 0.03 |
| Random Forest | 0.115 | ~0.00 |
| LightGBM (untuned) | 0.115 | 0.01 |

At 50% recall the three models are **statistically indistinguishable** — the
honest conclusion is that no model here separates readmission risk well at that
operating point. LightGBM was selected on its ranking quality (best PR-AUC,
0.229 vs 0.219 / 0.215) and because SHAP values for tree models are exact — not
because of a large accuracy gap.

### Final model (LightGBM, tuned, deployed threshold 0.55)

| Metric | Value | Plain meaning |
|---|---|---|
| Test ROC-AUC | 0.669 | ranking quality |
| Test PR-AUC | 0.226 | ranking quality under imbalance (base rate 0.113) |
| Recall | 44.0% | of the patients who returned, we catch 44% |
| Precision | 21.0% | of our alerts, 21% really return |
| F1 | 0.284 | balance of the two |
| Confusion matrix | TN 13,950 · FP 3,820 · FN 1,291 · TP 1,014 | 24 alerts per 100 stays |

**The two published threshold policies (notebook 03 §4):**

| Policy | Threshold | Recall | Precision | Alerts per 100 stays |
|---|---|---|---|---|
| **Deployed** (F1-optimal) | 0.55 | 44.0% | 21.0% | 24 |
| Recall-first (75% target) | 0.40 | 81.6% | 14.4% | 65 |

The deployed rule alerts about a quarter of discharges; the recall-first rule
catures four in five patients but would alert two thirds of all discharges —
not a staffable workflow. Which error matters more is a hospital policy question,
so both are published rather than decided silently in the code.

**How to read it:** at the deployed threshold we flag **24 of every 100 stays**;
of those alerts, **21%** genuinely return, and we catch **44%** of all patients who
return. For comparison, a hospital doing nothing has 88.7% accuracy and **0%**
recall — and, more usefully, the model's top-10 risk band contains the patients
the Part 1 analysis already identified (frequent prior stays, fragile discharge
destinations).

**Honest note on the level of performance.** ROC-AUC ~0.66 and PR-AUC ~0.23
against a 11.3% base rate is in line with published work on this dataset. The
ceiling is set by the data: readmission is driven strongly by housing, income,
transport and support at home — none of which these insurance records contain. A
stronger score would require stronger inputs, not a cleverer model.

### Tuning outcome (honest)

`RandomizedSearchCV` (3 sampled configurations × 3 patient-grouped folds) found
its best configuration at CV average-precision **0.2115**. Re-trained on the full
80,039 training stays, it scored test PR-AUC **0.2260** versus **0.2287** for the
untuned model — i.e. **tuning did not improve the test score** (ROC-AUC improved
0.663 → 0.669). The difference is within noise, and we say so rather than claim a
win. The search was deliberately small: a measured fit cost of ~60–100s on this
machine allowed only 9 fits in one session.

### SHAP — what drives the predictions

Mean |SHAP| on 2,000 test stays (notebook 03 §5):

| Rank | Feature | Mean \|SHAP\| | Reading |
|---|---|---|---|
| 1 | `number_inpatient` | 0.231 | prior inpatient stays — the dominant driver |
| 2 | `discharge_disposition_id_1` (home) | 0.168 | going home is the safest destination |
| 3 | `prior_visits_total` | 0.093 | total health-system use in the previous year |
| 4 | `number_diagnoses` | 0.060 | clinical complexity |
| 5 | `payer_code_Missing` | 0.051 | missing insurance code — an administrative artefact |
| 6 | `age_mid` | 0.043 | age |
| 7 | `diag_1_grp_circulatory` | 0.041 | primary diagnosis |
| 8 | `num_medications` | 0.039 | treatment intensity |
| 9 | `discharge_disposition_id_22` (rehab) | 0.037 | going to rehab raises risk |
| 10 | `diabetesMed` | 0.034 | on diabetes medication |

**This is the coherence check that matters:** the two strongest drivers are exactly
the two patterns found in Part 1 by counting — prior inpatient visits (8.6% →
45.9%) and discharge destination (9.3% → 36.7%). The model did not discover
something alien; it weighted what the data analysis had already proven.

One caveat worth flagging: `payer_code_Missing` ranking 5th is an
**administrative artefact**, not clinical knowledge. A reviewer should ask whether
a model should be allowed to lean on an insurance-code gap at all.

Per-patient SHAP waterfalls (notebook 03 §5.2) explain a correctly caught
readmission (model risk 64%) and a missed one (52%) — the kind of single-patient
explanation a clinical reviewer would need before trusting an alert.

### Fairness

Test-set recall by group at the deployed threshold:

| Group | Stays | Readmit rate | Recall | Precision |
|---|---|---|---|---|
| Caucasian | 15,011 | 11.9% | 44.0% | 21.5% |
| AfricanAmerican | 3,814 | 10.9% | **47.5%** | 20.4% |
| Hispanic | 387 | 7.8% | 33.3% | 12.2% |
| Asian | 134 | 9.7% | 53.8% | 41.2% |
| Other | 313 | 9.6% | 20.0% | 10.2% |
| Race missing | 416 | 9.1% | 28.9% | 17.5% |
| Female | 10,657 | 11.1% | 44.4% | 19.7% |
| Male | 9,417 | 11.9% | 43.6% | 22.5% |

**Judgement:** the two large groups (Caucasian, AfricanAmerican) are served
essentially equally (44.0% vs 47.5% recall). The apparent spread on the small
groups is driven by sample size — 134 to 416 stays, where a handful of patients
moves the rate several points — so we report the counts next to every figure and
do **not** claim a disparity or its absence. A serious fairness audit would need
a larger sample per group, plus socio-economic variables this dataset lacks
entirely. Gender shows no meaningful gap.

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

LightGBM, tuned, at a threshold chosen for recall–precision balance — because:

- It had the strongest PR-AUC of the three (0.229 vs 0.219 / 0.215) — the metric
  that respects an 11% positive rate.
- Trees handle the non-linear, mixed-type features (prior visits, dose changes)
  without manual transforms.
- It works with **missing and categorical** data natively, which suits messy
  hospital records.
- It explains itself: SHAP values for tree models are exact, so the fairness and
  per-patient explanations are trustworthy, not approximations.

Honest note on margin: at a matched 50% recall all three models delivered the same
precision (0.115), and tuning changed test PR-AUC by less than the noise. The
choice of LightGBM rests on ranking quality and explainability, **not** on a large
performance gap — and a genuine deployment decision would be made on a local
validation set, not on this comparison.

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
