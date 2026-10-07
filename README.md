# CY530 Assignment 2: DDoS Detection on CICDDoS2019 (Lynh Nguyen)

Random Forest (baseline and class-weighted) and optional XGBoost, for binary
(benign vs. attack) and multi-class (attack family) detection on CICDDoS2019 flow CSVs.

## 1. Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
Tested with Python 3.12 / scikit-learn 1.8. XGBoost is optional; the script skips it if missing.

## 2. Get the data
Download the CSV flow files (CICFlowMeter output) from
https://www.unb.ca/cic/datasets/ddos-2019.html and place all `.csv` files
under one folder, e.g. `data/`. The data is NOT included in this repo.
A Kaggle mirror of the full CSVs (about 30 GB) is at
https://www.kaggle.com/datasets/rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files
(requires a free Kaggle login). The script searches subfolders recursively, so keep the
original folder layout. Reading all files takes a while (expect tens of minutes to hours);
for a quick trial, copy a few CSVs into a smaller folder first.

## 3. Run (commands used for the paper; the data is a single CSV, cicddos2019_dataset.csv, from Mendeley Data, DOI 10.17632/ssnc74xm6r.1)
    python run_experiment.py --data data/ --out results --max-per-class 10000000 --seed 42
    python run_experiment.py --data data/ --out results_merged --max-per-class 10000000 --merge-families
    python ablation_features.py --data data/ --out results_ablation

Defaults: 70/30 stratified split, correlation filter at 0.95, top 30 features by
Random Forest importance. All feature selection is fit on the training split only.

## 4. Output (in `results/`)
- `summary_metrics.csv`: accuracy, macro P/R/F1, weighted F1, train/predict time per model and task
- `binary/` and `multiclass/`: per-model classification reports, confusion matrices (CSV + PNG)
- `feature_importance_{binary,multiclass}.csv`: top-ranked features
- `data_stats.json`: row counts after each cleaning step, class counts, run arguments

## 5. What the script does
1. Reads CSVs in chunks; caps each label at `--max-per-class` rows (seeded random sample).
2. Drops identifier columns (Flow ID, IPs, ports, timestamp), converts to numeric,
   replaces +/-inf with NaN, drops NaN rows and exact duplicates.
3. Splits train/test (stratified), then selects features on train only.
4. Trains and evaluates each model on the held-out test split.

## 6. Expected output
Numbers depend on the sampling cap and seed. Report the values from your own
`summary_metrics.csv`. Re-running with the same seed and data reproduces them.

## Credits
Baseline ideas adapted from the public repositories cited in the paper
(saghal/CIC-DDoS2019-ML-Detection; rakibnsajib/DDoS-Defense-...). Add a note here
on exactly what you reused versus rewrote.
