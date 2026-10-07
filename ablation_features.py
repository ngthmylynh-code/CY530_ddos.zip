#!/usr/bin/env python3
"""Feature-selection ablation: RF (100 trees, defaults) on ALL cleaned features vs. the selected top-30.
Same cleaning, split and seed as run_experiment.py. Usage: python ablation_features.py --data data/ --out results_ablation
"""
import argparse, os, time, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import run_experiment as rx

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results_ablation")
ap.add_argument("--seed", type=int, default=42); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
raw, _ = rx.load(a.data, 10**7, a.seed); df, _ = rx.clean(raw, a.seed, 10**7)
X, y_m = df.drop(columns="label"), df["label"]
y_b = (y_m.str.upper() != "BENIGN").map({False: "BENIGN", True: "ATTACK"})
rows = []
for task, y in [("binary", y_b), ("multiclass", y_m)]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=a.seed)
    sel, _ = rx.select_features(Xtr, ytr, 0.95, 30, a.seed)
    for name, cols in [("all_77_features", list(X.columns)), ("selected_30", sel)]:
        m = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=a.seed)
        t = time.time(); m.fit(Xtr[cols], ytr); fit_s = time.time() - t
        t = time.time(); p = m.predict(Xte[cols]); pred_s = time.time() - t
        pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="macro", zero_division=0)
        rows.append(dict(task=task, features=name, n_features=len(cols), accuracy=accuracy_score(yte, p),
                         macro_precision=pr, macro_recall=rc, macro_f1=f1, train_seconds=fit_s,
                         predict_us_per_flow=pred_s / len(yte) * 1e6)); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(a.out, "ablation_feature_selection.csv"), index=False)
