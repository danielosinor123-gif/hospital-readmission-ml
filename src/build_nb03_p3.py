"""
BUILDER for notebook 03, part 3 -- SHAP explainability, fairness, limitations,
final save.
"""
import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = PROJECT_ROOT / "notebooks" / "03_tuning_threshold_and_explainability.ipynb"

nb = nbf.read(NB_PATH, as_version=4)
cells = nb["cells"]

cells.append(nbf.v4.new_markdown_cell("""
## 5. Explainability — which features drive the predictions?

A model the hospital cannot question is a model the hospital will not use. We use
**SHAP** (SHapley Additive exPlanations):

- Every prediction starts from a baseline (the average risk, ~11%).
- Each feature **pushes the risk up or down** from that baseline.
- SHAP fairly splits the credit among all features.

For a tree model the SHAP values are exact and fast. We explain a random sample
of 2,000 test stays, then look at single patients.
"""))

cells.append(nbf.v4.new_code_cell("""# stratified sample of test stays for explanation (fast + representative)
shap_idx = pd.concat([
    y_test[y_test == 1].sample(1000, random_state=RANDOM_STATE),
    y_test[y_test == 0].sample(1000, random_state=RANDOM_STATE),
]).index
X_shap = X_test.loc[shap_idx]

explainer = shap.TreeExplainer(final_gbm)
shap_values = explainer.shap_values(X_shap, check_additivity=False)
print("shap values shape:", np.asarray(shap_values).shape,
      "| features:", X_shap.shape[1])"""))

cells.append(nbf.v4.new_markdown_cell("""
### 5.1 The overall picture — which features matter, and in which direction
"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 3: global importance ---------------------------------------------------
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=15)
plt.title("Figure 3 — Mean SHAP importance (global)")
plt.tight_layout()
plt.show()

print("Read it as: on average, how much each feature moved the risk prediction.")
print("Longer bar = more influence on the model's decisions.")"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 4: direction and distribution ---------------------------------------
shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
plt.title("Figure 4 — SHAP beeswarm: value (colour) vs impact")
plt.tight_layout()
plt.show()

print("How to read it:")
print("  each dot is ONE stay; x = how much that feature pushed the risk")
print("  colour = the feature's value: red = high, blue = low")
print("  e.g. if number_inpatient dots run red on the right, many prior stays")
print("  push the risk UP -- which matches our Chapter 1 finding.")"""))

cells.append(nbf.v4.new_markdown_cell("""
### 5.2 One patient at a time — the chart a nurse could be shown

A **waterfall** explains a single prediction: baseline → each feature's push →
final risk.
"""))

cells.append(nbf.v4.new_code_cell("""# pick one true positive (a patient the model caught) and one false negative
pred_all = (tuned_proba >= CHOSEN).astype(int)
tp_idx = y_test.index[(y_test == 1) & (pred_all == 1)][0]
fn_idx = y_test.index[(y_test == 1) & (pred_all == 0)][0]

for label, idx in [("a correctly caught readmission", tp_idx),
                   ("a missed readmission (what we want to understand)", fn_idx)]:
    x_one = X_test.loc[[idx]]
    sv_one = explainer.shap_values(x_one, check_additivity=False)
    st = (shap.Explanation(values=np.asarray(sv_one)[0],
                           base_values=explainer.expected_value,
                           data=x_one.iloc[0], feature_names=list(X_test.columns)))
    fig = shap.plots.waterfall(st, max_display=12, show=False)
    fig.set_size_inches(9, 6)
    plt.title(f"Figure 5 — {label}", fontsize=12)
    plt.tight_layout()
    plt.show()

    proba_one = final_gbm.predict_proba(x_one)[0, 1]
    print(f"model risk for this stay: {proba_one:.1%}  (threshold {CHOSEN:.2f})")
    print("-" * 70)"""))

cells.append(nbf.v4.new_markdown_cell("""
## 6. Fairness — does the model work equally well for everyone?

A model can be accurate overall and still fail one group. We break the test
results down by **race** and **gender** and compare recall: *"when a patient from
group X truly returned, how often did we alert?"*

This matters because Part 1 showed race/gender are barely predictive — but the
model could still perform worse on a minority group, and that is a fairness
failure even if the headline numbers look good.
"""))

cells.append(nbf.v4.new_code_cell("""# Reconstruct the demographic groups from the one-hot columns
race_cols = [c for c in X_test.columns if c.startswith("race_")]
def get_group(cols, prefix):
    sub = X_test[cols]
    labels = sub.idxmax(axis=1).str.replace(prefix, "", regex=False)
    labels[sub.sum(axis=1) == 0] = "unmapped"
    return labels

X_test["_race"] = get_group(race_cols, "race_")
X_test["_gender"] = np.where(X_test["gender"] == 0, "Female",
                             np.where(X_test["gender"] == 1, "Male", "Other"))

overall_recall = recall_score(y_test, final_pred)
print(f"overall recall: {overall_recall:.1%}")
print()
for group_col in ["_race", "_gender"]:
    stats = []
    for g, mask in X_test.groupby(group_col).groups.items():
        m = X_test.index.isin(mask)
        stats.append({
            "group": g,
            "stays": int(m.sum()),
            "readmit_rate": round(y_test[m].mean(), 3),
            "recall": round(recall_score(y_test[m], final_pred[m], zero_division=0), 3),
            "precision": round(precision_score(y_test[m], final_pred[m],
                                               zero_division=0), 3),
        })
    t = pd.DataFrame(stats).sort_values("stays", ascending=False)
    print(f"--- by {group_col.lstrip('_')} ---")
    print(t.to_string(index=False))
    print()"""))

cells.append(nbf.v4.new_code_cell("""# --- Chart 6: recall by group ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, col in zip(axes, ["_race", "_gender"]):
    t = (X_test.groupby(col)
             .apply(lambda d: pd.Series({
                 "stays": len(d),
                 "recall": recall_score(y_test.loc[d.index], final_pred[d.index],
                                        zero_division=0)}),
                    include_groups=False)
             .reset_index())
    t = t[t["stays"] >= 100]                     # groups big enough to judge
    t = t.sort_values("recall")
    sns.barplot(data=t, x=t.columns[0], y="recall", ax=ax, palette="muted")
    ax.axhline(overall_recall, color="red", ls="--", lw=1.5, label="overall recall")
    ax.set_ylim(0, 1)
    ax.set_title(f"Recall by {col.lstrip('_')} (groups with >=100 stays)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
plt.suptitle("Figure 6 — Does the model protect every group equally?", y=1.03)
plt.tight_layout()
plt.show()

print("If a bar sits well below the red line, the model is less protective for")
print("that group -- a fairness finding that must be reported, not averaged away.")"""))

cells.append(nbf.v4.new_markdown_cell("""
## 7. Limitations and why this must not touch a real ward

1. **The data is from 1999–2008.** Readmission patterns, treatments and coding
   practices have changed. A model trained on it is a historical artefact.
2. **One payer, one provider network.** 130 hospitals of one insurer: not
   necessarily representative of another hospital, state or country.
3. **Labels are imperfect.** "NO" lumps together "fine" and "returned but we
   never saw it" — including people who died or moved away.
4. **No socio-economic data.** Housing, transport, income and social support
   drive readmission strongly and are absent — so the model cannot see the real
   causes, only their shadows.
5. **Splits are random, not temporal.** The honest test is "train on 1999–2004,
   test on 2005–2008"; a random split is the easier version.
6. **Calibration is not guaranteed.** A 30% output does not automatically mean
   "30 out of 100 such patients return"; calibration must be verified per site.
7. **No clinical trial.** Lacking prospective validation, the model has zero
   evidence that acting on it helps patients — it may even cause harm by
   misdirecting scarce follow-up resources.

**Therefore:** this is a portfolio project demonstrating an honest ML pipeline,
not a clinical device.
"""))

cells.append(nbf.v4.new_code_cell("""# --- Save everything the report and the app need --------------------------------
MODELS = PROC.parent.parent / "models"
joblib.dump(final_gbm, MODELS / "final_model.joblib")

summary = {
    "model": "LightGBM (tuned on full training set)",
    "chosen_threshold": CHOSEN,
    "test_roc_auc": round(roc_auc_score(y_test, tuned_proba), 4),
    "test_pr_auc": round(average_precision_score(y_test, tuned_proba), 4),
    "test_recall": round(recall_score(y_test, final_pred), 4),
    "test_precision": round(precision_score(y_test, final_pred), 4),
    "test_f1": round(f1_score(y_test, final_pred), 4),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}
with open(PROC / "final_model_summary.json", "w") as f:
    json.dump(summary, f, indent=1)

print(json.dumps(summary, indent=1))
print("saved models/final_model.joblib and data/processed/final_model_summary.json")"""))

cells.append(nbf.v4.new_markdown_cell("""
---
**End of notebook 03.** All parts of the brief are now covered:
Part 1 → notebook 01 · Part 2 → notebook 02 §2–6 · Part 3 → notebooks 02–03 ·
Part 4 → notebook 03 §5–7 · Part 5 → `app/app.py` · Part 6 → `report.md`.
"""))

nb["cells"] = cells
nbf.write(nb, NB_PATH)
print("appended; notebook now has", len(cells), "cells ->", NB_PATH)
