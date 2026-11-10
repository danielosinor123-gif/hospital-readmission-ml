"""Time one LightGBM fit so the learning curve can be sized realistically."""
import time
from pathlib import Path
import pandas as pd
from lightgbm import LGBMClassifier

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
X_train = pd.read_csv(PROC / "X_train.csv")
y_train = pd.read_csv(PROC / "y_train.csv").squeeze()

print("train shape:", X_train.shape, "| positives:", int(y_train.sum()))
spw = (y_train == 0).sum() / (y_train == 1).sum()

for frac in (1.0, 0.5):
    n = int(len(X_train) * frac)
    X, y = X_train.iloc[:n], y_train.iloc[:n]
    m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                       scale_pos_weight=spw, n_jobs=-1, random_state=42, verbose=-1)
    t0 = time.time()
    m.fit(X, y)
    print(f"n={n:6d} rows -> fit took {time.time() - t0:6.1f}s")
