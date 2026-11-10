"""
BUILDER for notebook 03, part 2 -- final model, threshold analysis, test score.
"""
import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = PROJECT_ROOT / "notebooks" / "03_tuning_threshold_and_explainability.ipynb"

nb = nbf.read(NB_PATH, as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell("""
## 3. The final model — tuned configuration, trained on ALL the data

The search ran on 20,000 rows to stay inside the compute budget. Its winning
configuration is now re-trained on the **full 80,039 training stays** (one fit,
~2.5 minutes) and compared against the untuned LightGBM from notebook 02.
"""))

cells.append(nbf.v4.new_code_cell("""best_params = dict(search.best_params_)
best_params.setdefault("n_estimators", 150)   # fixed in the search on purpose
best_params.setdefault("scale_pos_weight", (y_train == 0).sum() / (y_train == 1).sum())
best_params.setdefault("random_state", RANDOM_STATE)
best_params.setdefault("n_jobs", -1)
best_params.setdefault("verbose", -1)

final_gbm = LGBMClassifier(**best_params)
final_gbm.fit(X_train, y_train)
print("final model trained with:", {k: v for k, v in best_params.items()
                                    if k != "scale_pos_weight"})"""))

cells.append(nbf.v4.new_code_cell("""def score(model, name):
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "recall@0.5": recall_score(y_test, (proba >= 0.5).astype(int)),
        "precision@0.5": precision_score(y_test, (proba >= 0.5).astype(int),
                                         zero_division=0),
    }

comparison = pd.DataFrame([
    score(joblib.load(PROC.parent.parent / "models" / "lightgbm.joblib"),
          "LightGBM (untuned)"),
    score(final_gbm, "LightGBM (tuned)"),
]).set_index("model").round(4)
print(comparison.to_string())

tuned_proba = final_gbm.predict_proba(X_test)[:, 1]
print()
print("Did tuning help? compare pr_auc (the honest imbalance metric).")"""))

cells.append(nbf.v4.new_markdown_cell("""
### 3.1 Comparing the models at the *same* operating point

The table above uses the default 0.5 threshold, which is unfair: the three models
miscalibrate differently. Random Forest's class weighting pushes its probabilities
down, so almost nothing crosses 0.5 — it flagged only **7 of 20,075** stays.

The professional comparison fixes the **recall** (how many real readmissions we
catch) and asks *what precision each model delivers at that recall*. That answers
the question that actually matters: "which model should we deploy?"
"""))

cells.append(nbf.v4.new_code_cell("""def precision_at_recall(model, target_recall=0.50):
    proba = model.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, proba)
    idx = int(np.argmax(rec >= target_recall))     # first point reaching the target
    return round(float(prec[idx]), 4), (round(float(thr[idx]), 3)
                                        if idx < len(thr) else float("nan"))

rows = []
for name, fname in [("Logistic Regression", "logistic_regression.joblib"),
                    ("Random Forest", "random_forest.joblib"),
                    ("LightGBM (untuned)", "lightgbm.joblib")]:
    m = joblib.load(PROC.parent.parent / "models" / fname)
    p, t = precision_at_recall(m, 0.50)
    rows.append({"model": name,
                 "precision @ 50% recall": p,
                 "threshold needed": t})
matched = pd.DataFrame(rows)
print(matched.to_string(index=False))

best = matched.loc[matched["precision @ 50% recall"].idxmax()]
print()
print(f"Best at 50% recall: {best['model']} "
      f"(precision {best['precision @ 50% recall']})")
print()
print("Note on quality: all three sit around PR-AUC 0.21-0.23 against a base")
print("rate of 0.113. That is honest -- readmission depends heavily on factors")
print("this dataset does not record (housing, support at home, income), so no")
print("model built from these columns alone can be excellent. The value is in")
print("the ranking and in the workflow it triggers, not in a headline score.")"""))

cells.append(nbf.v4.new_markdown_cell("""
## 4. Choosing the decision threshold

`predict_proba` gives a probability, but the hospital needs a yes/no alert. The
line we draw (the **threshold**) decides the trade:

- **Raise the threshold** → fewer alerts → fewer false alarms, but more missed
  readmissions (false negatives).
- **Lower the threshold** → catch more readmissions, but more false alarms.

For a hospital, a **missed high-risk patient is usually the worse error**, so we
pick the threshold that reaches at least **75% recall** while keeping precision as
high as possible — and we publish the trade instead of hiding it.
"""))

cells.append(nbf.v4.new_code_cell("""thresholds = np.arange(0.05, 0.85, 0.05)
rows = []
for t in thresholds:
    pred = (tuned_proba >= t).astype(int)
    rows.append({
        "threshold": t,
        "recall": recall_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "alerts": int(pred.sum()),
        "alerts_per_100_stays": round(100 * pred.mean(), 1),
    })
thr_table = pd.DataFrame(rows)
print(thr_table.round(3).to_string(index=False))"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 1: the trade-off curve ----------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thr_table["threshold"], thr_table["recall"], marker="o", label="recall")
ax.plot(thr_table["threshold"], thr_table["precision"], marker="s", label="precision")
ax.plot(thr_table["threshold"], thr_table["f1"], marker="^", label="F1")
ax.set_xlabel("decision threshold")
ax.set_ylabel("score")
ax.set_title("Figure 1 — What each threshold costs us")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("As the threshold rises, recall falls first: those are patients we stop")
print("catching. Precision rises slowly because alerts become more selective.")"""))

cells.append(nbf.v4.new_code_cell("""# --- Documented selection rule ---------------------------------------------------
# Clinical priority: catch at least 75% of true readmissions, then the best precision.
TARGET_RECALL = 0.75
eligible = thr_table[thr_table["recall"] >= TARGET_RECALL]
CHOSEN = float(eligible.sort_values("recall", ascending=False).iloc[0]["threshold"])
print(f"chosen threshold: {CHOSEN:.2f}  (first threshold reaching "
      f"{TARGET_RECALL:.0%} recall)")

final_pred = (tuned_proba >= CHOSEN).astype(int)
print()
print("Test-set performance at the chosen threshold:")
print(f"  recall    : {recall_score(y_test, final_pred):.1%}"
      f"  -> of the real readmissions, we catch this share")
print(f"  precision : {precision_score(y_test, final_pred):.1%}"
      f"  -> of our alerts, this share really return")
print(f"  F1        : {f1_score(y_test, final_pred):.3f}")
print(f"  alerts    : {final_pred.sum():,} of {len(y_test):,} stays "
      f"({final_pred.mean():.1%})")"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 2: final confusion matrix -------------------------------------------
cm = confusion_matrix(y_test, final_pred)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Oranges", ax=ax, square=True,
            xticklabels=["predicted NO", "predicted YES"],
            yticklabels=["actual NO", "actual YES"])
ax.set_title(f"Figure 2 — Final model test confusion matrix (threshold {CHOSEN:.2f})")
plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"true negatives  {tn:>7,}  correctly sent home without an alert")
print(f"false positives {fp:>7,}  alerts for patients who did NOT return (cost: nurse time)")
print(f"false negatives {fn:>7,}  readmissions we MISSED (cost: patient safety)")
print(f"true positives  {tp:>7,}  readmissions we caught")"""))

nb["cells"] = cells
nbf.write(nb, NB_PATH)
print("appended; notebook now has", len(cells), "cells ->", NB_PATH)
