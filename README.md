# CY530 Assignment 2: DDoS Detection on CICDDoS2019 (Lynh Nguyen)

Random Forest (default and class-weighted) and scikit-learn HistGradientBoosting for binary
(benign vs. attack) and 17-class (attack type) detection on a preprocessed CICDDoS2019 release.

## 1. Setup
    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
Tested with Python 3.12.3, scikit-learn 1.8.0, pandas 3.0.2, NumPy 2.4.4. XGBoost is optional
(the script skips it if missing; the paper does not use it).

## 2. Data access
Download `cicddos2019_dataset.csv` (431,371 rows, 80 columns) from Mendeley Data:
https://data.mendeley.com/datasets/ssnc74xm6r/1  (DOI 10.17632/ssnc74xm6r.1).
Create a folder `data/` next to the scripts and put the CSV inside. The data is not in this repo.

## 3. Run (about 10 minutes on one CPU core for the main run; 4 GB RAM is enough)
    python run_experiment.py --data data/ --out results --max-per-class 10000000 --seed 42
    python run_experiment.py --data data/ --out results_merged --max-per-class 10000000 --merge-families
    python ablation_features.py --data data/ --out results_ablation
`--max-per-class 10000000` keeps every row (no subsampling). Run all commands from the folder
containing the scripts.

## 4. Expected output (seed 42; tiny differences are possible across library versions)
| Task | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| Binary | RF baseline | 0.9994 | 0.9991 |
| Binary | RF class-weighted | 0.9993 | 0.9990 |
| Binary | HistGradBoost | 0.9995 | 0.9993 |
| 17-class | RF baseline | 0.9266 | 0.5222 |
| 17-class | RF class-weighted | 0.9262 | 0.5185 |
| 17-class | HistGradBoost | 0.9332 | 0.5206 |

`results_merged/` (DrDoS_X merged with X, 13 classes, baseline RF): accuracy 0.9735, macro-F1 0.6950.
`results_ablation/` (RF, all 77 features vs. selected 30): see paper Table 3.
Expected data statistics (`results/data_stats.json`): 431,371 raw rows, 425,910 after removing
5,461 duplicates; 298,137 train / 127,773 test flows.

## 5. What the script does
1. Streams the CSV and keeps a uniform random sample of at most `--max-per-class` rows per label.
2. Drops the row index and the `Class` column (it duplicates the binary target and would leak it),
   converts to numeric, removes inf/NaN rows, and merges the label spellings UDPLag and UDP-lag.
3. Removes exact duplicate rows BEFORE splitting, then makes a stratified 70/30 split.
4. Fits feature selection on the training split only: drop constants, drop one of each pair with
   |r| > 0.95, keep the top 30 by Random Forest importance.
5. Trains and evaluates each model; writes metrics, per-class reports, confusion matrices (CSV + PNG),
   feature importances, and `data_stats.json`.

## 6. Output files (in each results folder)
`summary_metrics.csv`, `binary/` and `multiclass/` (reports and confusion matrices),
`feature_importance_*.csv`, `data_stats.json`.

## 7. Baseline and changes
The baseline is the default-parameter Random Forest. Changes tested: class weighting
(`balanced_subsample`), gradient boosting, feature selection (ablation), and a post-hoc label merge.

## Credits
The pipeline design (cleaning, feature reduction, tree-ensemble baselines) was informed by
saghal/CIC-DDoS2019-ML-Detection and rakibnsajib/DDoS-Defense-A-Multiclass-and-Multidimensional-Detection-System-with-Diverse-Machine-Learning-Models.
No code was copied from either repository; all scripts were written for this project with AI assistance
(see the AI Use Disclosure in the paper).
