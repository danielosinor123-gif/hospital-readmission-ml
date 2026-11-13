"""
Compute the default value of every model feature and save it as a small JSON.

WHY THIS EXISTS
---------------
The Streamlit app asks the user for the ~20 clinically meaningful inputs and
fills the remaining ~200 features with the value seen most often in training
(the mode). Computing that from X_train.csv requires a 35 MB file that is NOT
committed to git, so the app could not run after a fresh clone.

This script runs ONCE on the machine that has X_train.csv and writes a ~40 KB
JSON the app can read anywhere (including Streamlit Cloud).

Run:  python src/make_feature_defaults.py
"""
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "data" / "processed"

X_train = pd.read_csv(PROC / "X_train.csv")
features = json.loads((PROC / "feature_columns.json").read_text())

defaults = {}
for c in features:
    mode = X_train[c].mode()
    defaults[c] = (int(mode.iloc[0]) if len(mode) else 0)   # mode; 0 if all missing

out = PROC / "feature_defaults.json"
out.write_text(json.dumps(defaults, indent=1))
print(f"wrote {out} ({out.stat().st_size:,} bytes, {len(features)} features)")
print("examples:", {k: defaults[k] for k in features[:3]})