"""
BUILDER PART 4 of notebook 02 -- overfitting investigation + save models.
"""
import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = PROJECT_ROOT / "notebooks" / "02_preprocessing_and_modeling.ipynb"

nb = nbf.read(NB_PATH, as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell("""
## 10. Overfitting — did the model memorise the training data?

**Overfitting** = the model learned the practice set (including its noise) so well
that it fails on unseen patients. The symptom is a **gap**: much better numbers on
the training pile than on the test pile.

Two checks below: (a) the same metrics computed on train vs test, and
(b) learning curves — how the score changes as the training set grows.
"""))

cells.append(nbf.v4.new_code_cell("""# --- (a) train vs test metrics, side by side -------------------------------------
def gap_table(model, name):
    tr = model.predict_proba(X_train)[:, 1]
    te = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "train_roc_auc": roc_auc_score(y_train, tr),
        "test_roc_auc": roc_auc_score(y_test, te),
        "gap": roc_auc_score(y_train, tr) - roc_auc_score(y_test, te),
    }

gaps = pd.DataFrame([gap_table(log_reg, "Logistic Regression"),
                     gap_table(rf, "Random Forest"),
                     gap_table(gbm, "LightGBM")]).set_index("model").round(4)
print(gaps.to_string())
print()
print("How to read 'gap':")
print("  ~0.00-0.01  -> generalises well")
print("  0.01-0.03   -> mild memorisation, acceptable, worth tuning")
print("  > 0.05      -> serious overfitting, must be fixed")"""))

cells.append(nbf.v4.new_code_cell("""# --- (b) learning curve for the strongest model --------------------------------
# COMPUTE REALITY CHECK: one full-size LightGBM fit takes ~2.5 minutes on this
# machine, so a 12-fit learning curve would run 30+ minutes. We therefore measure
# the curve on a stratified 20,000-row SUBSAMPLE with a lighter probe model.
# What we read is the SHAPE (the gap between the two lines), not the absolute
# scores -- and that shape is stable across model sizes.
from sklearn.model_selection import learning_curve, train_test_split

sub_X, _, sub_y, _ = train_test_split(
    X_train, y_train, train_size=20000, stratify=y_train, random_state=RANDOM_STATE
)

probe = LGBMClassifier(
    n_estimators=100, learning_rate=0.10, num_leaves=31,
    scale_pos_weight=(sub_y == 0).sum() / (sub_y == 1).sum(),
    n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
)

train_sizes, train_scores, test_scores = learning_curve(
    probe, sub_X, sub_y,
    train_sizes=[0.3, 0.6, 1.0],
    cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
    scoring="roc_auc",
    n_jobs=1,
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="training score")
ax.plot(train_sizes, test_scores.mean(axis=1), marker="s", label="cross-validation score")
ax.fill_between(train_sizes,
                test_scores.mean(axis=1) - test_scores.std(axis=1),
                test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.2)
ax.set_xlabel("number of training stays")
ax.set_ylabel("ROC-AUC")
ax.set_title("Figure 3 — Learning curve (LightGBM)")
ax.legend()
plt.tight_layout()
plt.show()

print("The gap between the two lines is the overfitting gap.")
print("If the lines converge as data grows, more data is the cure.")
print("If they stay apart, the model needs simplification (fewer leaves, more")
print("regularisation) -- exactly what notebook 03 does with tuning.")"""))

cells.append(nbf.v4.new_markdown_cell("""
**An honest note on the curve:** the folds here are ordinary 5-fold splits, so a
patient with several stays can still appear in different folds, which slightly
flatters the scores. Notebook 03 switches to `GroupKFold` (folds grouped by
patient) for the hyperparameter search, which is the rigorous version.
"""))

cells.append(nbf.v4.new_code_cell("""# --- Save the trained models and the metrics for the app and the report ----------
MODELS = DATA / "models"
MODELS.mkdir(exist_ok=True)

joblib.dump(log_reg, MODELS / "logistic_regression.joblib")
joblib.dump(rf, MODELS / "random_forest.joblib")
joblib.dump(gbm, MODELS / "lightgbm.joblib")

metrics_out = table.reset_index().to_dict(orient="records")
with open(PROC / "model_metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=1)

print("saved models:", [p.name for p in MODELS.glob('*.joblib')])
print("saved metrics to model_metrics.json")"""))

cells.append(nbf.v4.new_markdown_cell("""
---
**End of notebook 02.** What is deliberately left for `03_tuning_and_explainability.ipynb`:
- `GroupKFold` cross-validation by patient,
- hyperparameter tuning with `RandomizedSearchCV`,
- threshold choice (how many false positives do we accept?),
- SHAP explanations,
- and the healthcare / fairness / limitations discussion.

**What you should be able to say about this notebook:**
> "I split 100,114 hospital stays into ~80k training and ~20k test, grouped by
> patient so nobody appears on both sides. Three models were trained on the
> training pile only and compared on the test pile with precision, recall, F1,
> ROC-AUC and PR-AUC, because accuracy (88.7% for always answering 'no') is
> meaningless at an 11% positive rate."
"""))

nb["cells"] = cells
nbf.write(nb, NB_PATH)
print("appended; notebook now has", len(cells), "cells ->", NB_PATH)
