# PRVO TESTIRANJE — Diploma Project (FRI)

## Purpose
Benchmarking 6 classical/foundation-model tabular ML algorithms on OpenML
datasets as part of a diploma thesis at FRI (Faculty of Computer and
Information Science, Ljubljana), with default hyperparameters throughout.
Comments and docstrings in the codebase are written in Slovenian. Currently
3 datasets; designed to extend to ~72 datasets later without restructuring.

## Structure
- `config.yaml` — single source of config: dataset OpenML IDs, `n_splits`,
  `random_state`, algorithm list, output paths.
- `src/data.py` — `load_dataset(openml_id, ...)` downloads/caches a dataset
  from OpenML and builds the 5-fold `StratifiedKFold` splits **once**; every
  algorithm reuses the same fold indices for a given dataset.
- `src/utils.py` — `compute_roc_auc()`, binary vs. multiclass aware.
- `src/models/` — one file per algorithm, each exposing a uniform
  `run(X_train, y_train, X_test, y_test, categorical_cols) -> dict` returning
  `{"model", "roc_auc", "train_time_s", "inference_time_s", "error",
  "preprocessing", "raw_error"}`. `src/models/__init__.py` exposes `REGISTRY`
  mapping config algorithm names to `run` functions.
  Implemented: `random_forest.py`, `xgboost_model.py`, `lightgbm_model.py`,
  `catboost_model.py`, `tabpfn_model.py`, `tabicl_model.py`.
- `src/run_benchmark.py` — orchestrates: load each dataset, run every
  algorithm on every fold, write `results/results.csv` and
  `results/preprocessing_log.md`. Run via `python -m src.run_benchmark`.
- `src/summary.py` — prints mean ROC-AUC ± std per (dataset, algorithm) and
  mean ROC-AUC ± std / mean rank per algorithm across datasets. Run via
  `python -m src.summary`.
- `results/` — `results.csv` (columns: dataset, algorithm, fold, roc_auc,
  train_time_s, inference_time_s) and `preprocessing_log.md` (what
  preprocessing each algorithm × dataset combo required, and any raw-input
  errors). Both are generated, not hand-edited.
- `data/openml_cache/` — OpenML's local dataset cache (gitignored).

## Preprocessing policy (per algorithm — this is a deliberate experimental variable)
- **RandomForest**: no native NaN/categorical support → median imputation
  (numeric) + most-frequent imputation & ordinal encoding (categorical),
  fit on train fold only.
- **XGBoost**: native NaN handling; categorical columns ordinal-encoded to
  numeric codes with NaN preserved (`OrdinalEncoder(encoded_missing_value=np.nan)`).
- **LightGBM**: native NaN handling; categorical columns cast to pandas
  `category` dtype for native categorical support. Non-categorical columns
  are explicitly cast to float, since some nominally-numeric OpenML columns
  (e.g. an all-missing column) load as `object` dtype and LightGBM rejects
  non-numeric/category dtypes.
- **CatBoost**: native NaN handling (numeric) + native categorical support
  via `cat_features`; categorical columns cast to `str` first (CatBoost
  disallows float NaN inside categorical columns — casting makes `'nan'` its
  own category, not an imputation).
- **TabPFN / TabICL**: pass data as raw as possible by design — the
  experiment is to see what each model handles natively. Each model's
  `run()` tries the raw fold first; only on an exception does it apply a
  logged minimal fix and retry:
  - TabPFN: raw works except when a column is `object`-dtype with unparsable
    values (e.g. an all-missing OpenML column) → falls back to ordinal-encoding
    categorical columns (NaN preserved).
  - TabICL: raw data itself is never the problem; `device='auto'` fails to
    resolve a torch device index in this WSL2/CUDA setup → falls back to an
    explicit `device='cuda:0'` (or `'cpu'` if no GPU).
  Never silently preprocess — every raw failure + the fix applied is
  recorded in `results/preprocessing_log.md`.

## Environment
- Runs under WSL2 (Ubuntu) on Windows 11; working directory is on the
  Windows filesystem at `/mnt/d/fri/Diplomska/prvo_testiranje`.
- Conda env: `tabular` (Python 3.10). GPU: NVIDIA RTX 3060, `torch` 2.4.1+cu121,
  `torch.cuda.is_available()` is `True`. Run project scripts with
  `conda run -n tabular python -m src.<module>` (or activate the env first).
- `src/data.py` resolves the OpenML cache path relative to the repo root
  (`data/openml_cache`), not hardcoded — safe if the repo is moved.

## Datasets (OpenML IDs, set in `config.yaml`)
- 31 — credit-g (1000 rows, 20 attrs, 13 categorical, no missing values)
- 37 — diabetes / Pima (768 rows, 8 attrs, all numeric, no missing values)
- 38 — sick (3772 rows, 29 attrs, 22 categorical, has missing values).
  **Note**: the originally-specified ID 3021 does not exist on OpenML
  ("Unknown dataset"); 38 is the standard "sick" thyroid dataset and was
  confirmed with the user as the intended substitute.

## Conventions
- Each model module fails soft: exceptions are caught and stored in
  `result["error"]` rather than raised, so one failing (dataset, algorithm,
  fold) combo doesn't crash the full benchmark run.
- `random_state=42` used consistently for reproducibility (data split, CV
  folds, and every model's `random_state`).
- All algorithms use **default hyperparameters** — this is intentional per
  the thesis protocol, not an oversight to "fix".
