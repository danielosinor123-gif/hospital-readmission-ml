"""Diagnose the threshold-choice logic from notebook 03 without retraining."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import recall_score, precision_score

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
X_test = pd.read_csv(PROC / "X_test.csv")
y_test = pd.read_csv(PROC / "y_test.csv").squeeze().reset_index(drop=True)
model = joblib.load(PROC.parent.parent / "models" / "lightgbm.joblib")

proba = model.predict_proba(X_test)[:, 1]
print("probability range:", round(proba.min(), 4), "to", round(proba.max(), 4))
print("share >= 0.05:", round((proba >= 0.05).mean(), 4))
print()

for t in np.arange(0.05, 0.85, 0.05):
    pred = (proba >= t).astype(int)
    print(f"t={t:.2f}  recall={recall_score(y_test, pred, zero_division=0):.3f}  "
          f"precision={precision_score(y_test, pred, zero_division=0):.3f}  "
          f"flagged={pred.sum():>6,}")

# now the notebook's selection rule
thresholds = np.arange(0.05, 0.85, 0.05)
recalls = [recall_score(y_test, (proba >= t).astype(int), zero_division=0) for t in thresholds]
thr_table = pd.DataFrame({"threshold": thresholds, "recall": recalls})
eligible = thr_table[thr_table["recall"] >= 0.75]
print()
print("eligible (recall >= 0.75):", len(eligible))
if len(eligible):
    chosen = float(eligible.sort_values("recall", ascending=False).iloc[0]["threshold"])
    print("CHOSEN threshold:", chosen)
    pred_all = (proba >= chosen).astype(int)
    tp = y_test.index[(y_test == 1) & (pred_all == 1)]
    fn = y_test.index[(y_test == 1) & (pred_all == 0)]
    print("true positives:", len(tp), "| false negatives:", len(fn))
