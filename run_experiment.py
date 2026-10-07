#!/usr/bin/env python3
"""CY530 Assignment 2: Random Forest / XGBoost DDoS detection on CICDDoS2019 flow CSVs.

Usage:
    python run_experiment.py --data /path/to/csvs --out results --max-per-class 50000

Everything reported in the paper comes from files this script writes to --out.
"""
import argparse, glob, json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

DROP_COLS = ["flow id", "source ip", "destination ip", "source port", "destination port",
             "timestamp", "unnamed: 0", "simillarhttp", "inbound",
             "class"]  # identifiers / row index / duplicate Attack-Benign column (label leak)

# Same attack appears under two spellings in the preprocessed CSV
LABEL_FIX = {"UDPLag": "UDP-lag"}


def cap(df, n, seed):
    """Keep at most n random rows per label (shuffle, then take first n of each group)."""
    return df.sample(frac=1, random_state=seed).groupby("label").head(n)


def load(data_dir, max_per_class, seed):
    """Stream all CSVs and keep a uniform random sample of at most max_per_class rows per label.

    Each row gets a random key; after every chunk we keep the rows with the smallest keys per
    label. This is an exact uniform sample and memory stays bounded (~labels x max_per_class).
    """
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"No CSV files found under {data_dir}")
    rng = np.random.default_rng(seed)
    kept = None
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] reading {f}", flush=True)
        for chunk in pd.read_csv(f, chunksize=200_000, low_memory=False):
            chunk.columns = chunk.columns.str.strip().str.lower()
            chunk["label"] = chunk["label"].astype(str).str.strip()
            chunk["_k"] = rng.random(len(chunk))
            kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
            kept = kept.sort_values("_k").groupby("label").head(max_per_class)
    kept = kept.drop(columns="_k").reset_index(drop=True)
    return kept, len(files)


def clean(df, seed, max_per_class, merge_families=False):
    stats = {"rows_raw": len(df)}
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df["label"] = df["label"].astype(str).str.strip().replace(LABEL_FIX)
    if merge_families:  # post-hoc analysis: DrDoS_UDP and UDP become one class, etc.
        df["label"] = df["label"].str.replace("^DrDoS_", "", regex=True)
    num = df.drop(columns="label").apply(pd.to_numeric, errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan)
    df = pd.concat([num, df["label"]], axis=1)
    df = df.dropna()
    stats["rows_after_inf_nan"] = len(df)
    df = df.drop_duplicates()
    stats["rows_after_dedup"] = len(df)
    # re-cap per class after concatenation across files
    df = cap(df, max_per_class, seed)
    stats["rows_final"] = len(df)
    stats["class_counts"] = df["label"].value_counts().to_dict()
    return df.reset_index(drop=True), stats


def select_features(Xtr, ytr, corr_thr, top_k, seed):
    """All selection is fit on the TRAINING split only (no leakage)."""
    keep = [c for c in Xtr.columns if Xtr[c].nunique() > 1]          # drop constants
    corr = Xtr[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    keep = [c for c in keep if not any(upper[c] > corr_thr)]          # drop highly correlated
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=seed)
    rf.fit(Xtr[keep], ytr)
    imp = pd.Series(rf.feature_importances_, index=keep).sort_values(ascending=False)
    return list(imp.index[:top_k]), imp


def evaluate(name, model, Xtr, ytr, Xte, yte, labels, out):
    t0 = time.time(); model.fit(Xtr, ytr); fit_s = time.time() - t0
    t0 = time.time(); pred = model.predict(Xte); pred_s = time.time() - t0
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    _, _, wf1, _ = precision_recall_fscore_support(yte, pred, average="weighted", zero_division=0)
    res = {"model": name, "accuracy": accuracy_score(yte, pred), "macro_precision": p,
           "macro_recall": r, "macro_f1": f1, "weighted_f1": wf1,
           "train_seconds": fit_s, "predict_seconds": pred_s,
           "predict_us_per_flow": pred_s / len(yte) * 1e6}
    with open(os.path.join(out, f"report_{name}.txt"), "w") as fh:
        fh.write(classification_report(yte, pred, zero_division=0, digits=4))
    cm = confusion_matrix(yte, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(out, f"confusion_{name}.csv"))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True).clip(min=1), cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f"Row-normalised confusion matrix: {name}")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"confusion_{name}.png"), dpi=200); plt.close(fig)
    return res


def get_models(seed):
    models = {
        "RF_baseline": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
        "RF_class_weight": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed,
                                                  class_weight="balanced_subsample"),
    }
    from sklearn.ensemble import HistGradientBoostingClassifier
    models["HistGradBoost"] = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    try:
        from xgboost import XGBClassifier  # optional
        models["XGBoost"] = ("xgb", XGBClassifier(n_estimators=200, max_depth=8, n_jobs=-1,
                                                  tree_method="hist", random_state=seed))
    except ImportError:
        print("xgboost not installed; skipping XGBoost.")
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-per-class", type=int, default=50_000)
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--corr-threshold", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge-families", action="store_true",
                    help="post-hoc: merge DrDoS_X and X labels; skips the binary task")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    np.random.seed(a.seed)

    raw, n_files = load(a.data, a.max_per_class, a.seed)
    df, stats = clean(raw, a.seed, a.max_per_class, a.merge_families)
    import sklearn
    stats["csv_files"] = n_files
    stats["versions"] = {"pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__}
    stats["n_features_before_selection"] = df.shape[1] - 1
    print(json.dumps(stats, indent=2))

    X, y_multi = df.drop(columns="label"), df["label"]
    y_bin = (y_multi.str.upper() != "BENIGN").astype(int).map({0: "BENIGN", 1: "ATTACK"})

    all_results = []
    tasks = [("multiclass", y_multi)] if a.merge_families else [("binary", y_bin), ("multiclass", y_multi)]
    for task, y in tasks:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=a.test_size, stratify=y,
                                              random_state=a.seed)
        feats, imp = select_features(Xtr, ytr, a.corr_threshold, a.top_k, a.seed)
        imp.head(a.top_k).to_csv(os.path.join(a.out, f"feature_importance_{task}.csv"),
                                 header=["importance"])
        labels = sorted(y.unique())
        out_t = os.path.join(a.out, task); os.makedirs(out_t, exist_ok=True)
        for name, m in get_models(a.seed).items():
            ytr_fit, yte_fit, lab = ytr, yte, labels
            if isinstance(m, tuple):  # XGBoost needs integer labels
                idx = {l: i for i, l in enumerate(labels)}
                m = m[1]; ytr_fit = ytr.map(idx); yte_fit = yte.map(idx); lab = list(range(len(labels)))
            r = evaluate(name, m, Xtr[feats], ytr_fit, Xte[feats], yte_fit, lab, out_t)
            r.update(task=task, n_train=len(Xtr), n_test=len(Xte), n_features=len(feats))
            all_results.append(r); print(r)

    pd.DataFrame(all_results).to_csv(os.path.join(a.out, "summary_metrics.csv"), index=False)
    with open(os.path.join(a.out, "data_stats.json"), "w") as fh:
        json.dump({**stats, "args": vars(a)}, fh, indent=2, default=str)
    print("Done. See", a.out)


if __name__ == "__main__":
    main()
