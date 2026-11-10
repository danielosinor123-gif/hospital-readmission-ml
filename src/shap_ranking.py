"""Compute the SHAP feature ranking for the final model (for the report)."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
X_test = pd.read_csv(PROC / "X_test.csv")
y_test = pd.read_csv(PROC / "y_test.csv").squeeze().reset_index(drop=True)

model = joblib.load(PROC.parent.parent / "models" / "final_model.joblib")

rng = np.random.default_rng(42)
idx = rng.choice(len(X_test), 2000, replace=False)
X = X_test.iloc[idx]

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X, check_additivity=False)

importance = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": np.abs(sv).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)

print(importance.head(15).round(5).to_string(index=False))
print()
print("baseline (expected) risk:", round(float(explainer.expected_value), 4))
