"""
BUILDER PART 3 of notebook 02 -- the three models + evaluation + charts.
"""
import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = PROJECT_ROOT / "notebooks" / "02_preprocessing_and_modeling.ipynb"

nb = nbf.read(NB_PATH, as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell("""
## 8. Three models — and why each one

| Model | Why we chose it | How it handles the imbalance |
|---|---|---|
| **Logistic Regression** | The honest baseline. Fast, hard to overfit, and its coefficients are directly interpretable ("each unit of X multiplies the odds by Y"). If a fancy model cannot beat this, the fancy model has not earned its place. | `class_weight="balanced"` |
| **Random Forest** | Hundreds of decision trees voting. Handles non-linear relationships (like prior visits) and mixed data types without much preparation. A strong, forgiving middle-ground. | `class_weight="balanced_subsample"` |
| **LightGBM** | Gradient-boosted trees — the modern workhorse on tabular data, usually the strongest of the three here. Builds trees one at a time, each fixing the previous one's mistakes. | `scale_pos_weight` = negatives / positives |

All three get the *same* features and the *same* train/test split, so the
comparison is fair.
"""))

cells.append(nbf.v4.new_code_cell("""# Which columns need scaling? (Logistic Regression only.)
# One-hot columns are already 0/1; the rest are genuine counts.
numeric_cols = [c for c in FEATURES if set(df_encoded[c].unique()) - {0, 1}]
binary_cols = [c for c in FEATURES if c not in numeric_cols]
print("numeric columns:", len(numeric_cols), "| binary/one-hot columns:", len(binary_cols))
print("harmless if a few 0/1-looking columns land in numeric: scaling 0/1 leaves them 0/1")"""))

cells.append(nbf.v4.new_code_cell("""# --- Model 1: Logistic Regression ----------------------------------------------
# Steps: scale the numeric columns -> feed to logistic regression.
# Wrapped in a Pipeline so scaling is LEARNED ON TRAIN ONLY (never on test).
log_reg = Pipeline([
    ("scale", ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("bin", "passthrough", binary_cols),
    ])),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=RANDOM_STATE)),
])

import time
t0 = time.time()
log_reg.fit(X_train, y_train)
print(f"trained in {time.time() - t0:.1f}s")
print("classes it learned:", log_reg.named_steps["clf"].classes_)"""))

cells.append(nbf.v4.new_code_cell("""# --- Model 2: Random Forest ----------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    n_jobs=-1,                      # use all CPU cores
    random_state=RANDOM_STATE,
)

t0 = time.time()
rf.fit(X_train, y_train)
print(f"trained in {time.time() - t0:.1f}s")"""))

cells.append(nbf.v4.new_code_cell("""# --- Model 3: LightGBM ----------------------------------------------------------
# scale_pos_weight tells the model: "treat each positive as if it were N copies",
# because there are 8.8 times more negatives than positives.
spw = (y_train == 0).sum() / (y_train == 1).sum()

gbm = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    scale_pos_weight=spw,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=-1,                    # silence the per-round logging
)

t0 = time.time()
gbm.fit(X_train, y_train)
print(f"trained in {time.time() - t0:.1f}s  (scale_pos_weight = {spw:.2f})")"""))

cells.append(nbf.v4.new_markdown_cell("""
## 9. Evaluate on the untouched test pile

The metrics, in plain language:

| Metric | Plain meaning | Why it matters here |
|---|---|---|
| **Recall** | of the patients who really came back, how many did we flag? | the clinical priority — a missed readmission is a sick patient at home |
| **Precision** | of the patients we flagged, how many really came back? | the cost of an intervention: nurses' time, a follow-up call, a bed |
| **F1** | the balance of the two above | one number to compare models |
| **ROC-AUC** | how well the model *ranks* patients (ignoring any threshold) | threshold-independent quality |
| **Average precision (PR-AUC)** | the same, but honest under imbalance | ROC-AUC flatters models on rare positives — PR-AUC does not |
"""))

cells.append(nbf.v4.new_code_cell("""def evaluate(model, name, X_tr, y_tr, X_te, y_te, threshold=0.5):
    \"\"\"Train on train, judge on test. Returns a dict of metrics.\"\"\"
    proba = model.predict_proba(X_te)[:, 1]         # probability of class 1
    pred = (proba >= threshold).astype(int)

    return {
        "model": name,
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "flagged": int(pred.sum()),                 # how many patients we alert on
        "proba": proba,
        "pred": pred,
    }

results = [
    evaluate(log_reg, "Logistic Regression", X_train, y_train, X_test, y_test),
    evaluate(rf, "Random Forest", X_train, y_train, X_test, y_test),
    evaluate(gbm, "LightGBM", X_train, y_train, X_test, y_test),
]

table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("proba", "pred")}
                      for r in results]).set_index("model").round(4)
print(table.to_string())
print()
print("Baseline to beat (always say NO): accuracy 88.7%, recall 0%")"""))

cells.append(nbf.v4.new_markdown_cell("""
### 9.1 Confusion matrices — what the model got right and wrong
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, r in zip(axes, results):
    cm = confusion_matrix(y_test, r["pred"])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax, square=True,
                xticklabels=["predicted NO", "predicted YES"],
                yticklabels=["actual NO", "actual YES"])
    ax.set_title(f"{r['model']}\\nrows = truth, columns = prediction")
plt.suptitle("Figure 1 — Confusion matrices on the test set", y=1.03, fontsize=13)
plt.tight_layout()
plt.show()

print("Read the bottom-left cell: patients who DID come back and we MISSED them.")
print("That number is the human cost of a false negative.")"""))

cells.append(nbf.v4.new_code_cell("""# --- ROC and Precision-Recall curves -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for r in results:
    fpr, tpr, _ = roc_curve(y_test, r["proba"])
    axes[0].plot(fpr, tpr, lw=2, label=f"{r['model']} ({r['roc_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="random guessing")
axes[0].set_xlabel("False positives (healthy people flagged)")
axes[0].set_ylabel("True positives (readmissions caught)")
axes[0].set_title("ROC curve — higher is better")
axes[0].legend()

base_rate = y_test.mean()
for r in results:
    prec, rec, _ = precision_recall_curve(y_test, r["proba"])
    axes[1].plot(rec, prec, lw=2, label=f"{r['model']} ({r['pr_auc']:.3f})")
axes[1].axhline(base_rate, color="grey", ls="--", lw=1,
                label=f"no-skill baseline ({base_rate:.3f})")
axes[1].set_xlabel("Recall (readmissions caught)")
axes[1].set_ylabel("Precision (of those flagged, how many real)")
axes[1].set_title("Precision-Recall curve — the honest chart under imbalance")
axes[1].legend()

plt.suptitle("Figure 2 — Model separation on the test set", y=1.03, fontsize=13)
plt.tight_layout()
plt.show()"""))

nb["cells"] = cells
nbf.write(nb, NB_PATH)
print("appended; notebook now has", len(cells), "cells ->", NB_PATH)
