"""Measure LightGBM fit time at the scale the search would actually use."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
X = pd.read_csv(PROC / "X_train.csv")
y = pd.read_csv(PROC / "y_train.csv").squeeze()

spw = (y == 0).sum() / (y == 1).sum()

rng = np.random.default_rng(42)
pos = X.index[y == 1].to_numpy()
neg = X.index[y == 0].to_numpy()
take = np.concatenate([rng.choice(pos, int(10000 * y.mean()), replace=False),
                       rng.choice(neg, 10000 - int(10000 * y.mean()), replace=False)])
Xs, ys = X.loc[take], y.loc[take]

for n_est in (100, 150):
    m = LGBMClassifier(n_estimators=n_est, learning_rate=0.1, num_leaves=31,
                       scale_pos_weight=spw, n_jobs=-1, random_state=42, verbose=-1)
    t0 = time.time()
    m.fit(Xs, ys)
    print(f"10,000 rows x {n_est} trees -> {time.time() - t0:.1f}s")
