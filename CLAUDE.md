# PRVO TESTIRANJE — Diploma Project (FRI)

## Purpose
Benchmarking classical ML models on the OpenML **credit-g** dataset (German Credit,
binary classification, dataset ID 31) as part of a diploma thesis at FRI (Faculty of
Computer and Information Science, Ljubljana). Comments and docstrings in the codebase
are written in Slovenian.

## Structure
- `data/loader.py` — downloads/caches `credit-g` from OpenML, produces stratified
  train/test split (80/20) and a 5-fold `StratifiedKFold` CV object via `load_dataset()`.
- `models/` — one file per model, each exposing a uniform
  `run(X_train, y_train, X_test, y_test) -> dict` interface returning
  `{"model", "roc_auc", "accuracy", "train_time_s", "inf_time_s", "error"}`.
  Implemented so far: `random_forest.py`, `xgboost_model.py`, `lightgbm_model.py`.
- `results/` — currently empty; intended output location for benchmark results.
- `main.py` — currently empty; expected to orchestrate loading data and running all
  models, likely writing to `results/`.
- `requirements.txt` — lists more than is currently used: `catboost`, `tabpfn`, and
  `torch`/`torchvision` are pinned but have no corresponding model module yet, so more
  models are planned.

## Environment
- Runs under WSL2 (Ubuntu) on Windows 11; working directory is on the Windows filesystem
  at `/mnt/d/fri/Diplomska/PRVO TESTIRANJE`.
- Not yet a git repository.
- `data/loader.py` hardcodes the absolute cache path
  `/mnt/d/fri/Diplomska/PRVO TESTIRANJE/data` for `openml.config.cache_directory` —
  keep this in mind if the project is ever moved or run outside this exact path.

## Conventions
- Each model module fails soft: exceptions are caught and stored in `result["error"]`
  rather than raised, so a single failing model doesn't crash a full benchmark run.
- `random_state=42` used consistently for reproducibility.
