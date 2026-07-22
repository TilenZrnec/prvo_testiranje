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
  - TabPFN (v3, `tabpfn` 8.x): raw input works on all current datasets,
    including `object`-dtype columns with unparsable/all-missing values
    (v2 needed an ordinal-encoding fallback there — kept as a safety net
    for future datasets, NaN preserved).
  - TabICL: raw input works. Under torch 2.4.1, `device='auto'` failed to
    resolve a torch device index in this WSL2/CUDA setup and a fallback to
    explicit `device='cuda:0'` was needed; fixed by the torch 2.13 upgrade,
    fallback kept as a safety net.
  Never silently preprocess — every raw failure + the fix applied is
  recorded in `results/preprocessing_log.md`.

## Environment
- Runs under WSL2 (Ubuntu) on Windows 11; working directory is on the
  Windows filesystem at `/mnt/d/fri/Diplomska/prvo_testiranje`.
- Conda env: `tabular` (Python 3.10). GPU: NVIDIA RTX 3060, `torch` 2.13.0+cu130,
  `torch.cuda.is_available()` is `True`. Run project scripts with
  `conda run -n tabular python -m src.<module>` (or activate the env first).
- TabPFN v3 (`tabpfn` 8.x) requires a one-time license acceptance via a
  PriorLabs account; the credential is cached locally on this machine. On a
  fresh machine: interactive first `fit()` opens a browser login (needs a
  real TTY — `conda activate`, not `conda run`), or set `TABPFN_TOKEN` from
  https://ux.priorlabs.ai/account for headless use.
- `src/data.py` resolves the OpenML cache path relative to the repo root
  (`data/openml_cache`), not hardcoded — safe if the repo is moved.

## Datasets (OpenML IDs, set in `config.yaml`)
- 31 — credit-g (1000 rows, 20 attrs, 13 categorical, no missing values)
- 37 — diabetes / Pima (768 rows, 8 attrs, all numeric, no missing values)
- 38 — sick (3772 rows, 29 attrs, 22 categorical, has missing values).
  **Note**: the originally-specified ID 3021 does not exist on OpenML
  ("Unknown dataset"); 38 is the standard "sick" thyroid dataset and was
  confirmed with the user as the intended substitute.

## Arnes HPC
- The benchmark also runs on the Arnes SLURM cluster (GPU partition `gpu`,
  H100 nodes). Env: micromamba prefix at `~/envs/tabular` (micromamba binary
  at `~/bin/micromamba`, no shell activation hooks in batch scripts).
- TabPFN token lives in `~/.tabpfn_token` (sourced by the batch script);
  compute nodes run offline (`HF_HUB_OFFLINE=1`), so run
  `python scripts/prestage.py` on the login node first — it caches OpenML
  datasets 31/37/38 and the TabPFN/TabICL weights (TabICL on CPU).
- `src/run_one_dataset.py --index N --ids-file scripts/subset_ids.json` runs
  all REGISTRY algorithms on the Nth OpenML ID and writes
  `results/per_dataset/<openml_id>.csv`; exits early ("already done") if the
  file exists, so re-submitted arrays skip finished datasets.
- Submit from the repo root: `sbatch scripts/run_subset.sh` (array 0-2; fill
  `--account`/`--reservation` from `sacctmgr show assoc user=$USER`).
- `scripts/merge_results.py <out.csv> [--input-dir DIR]` concatenates
  per-dataset CSVs (default `results/per_dataset/`);
  `scripts/compare_results.py <local.csv> <arnes.csv>` diffs ROC-AUC.

### Operational lessons (learned 2026-07-22, validation run)
- **The cluster remote uses SSH**, not HTTPS: `git@github.com:...`. The
  cluster has no credential helper for HTTPS, so an `https://` remote
  prompts for a password and fails in a batch context.
- **Scratch vs. curated results.** `results/per_dataset/` is *scratch*: it is
  gitignored, and `run_one_dataset.py` treats an existing
  `<openml_id>.csv` there as "already done" and skips the dataset. Therefore
  **`rm -rf results/per_dataset/*` before any real run** — otherwise stale
  files silently suppress recomputation. This actually happened: a
  git-committed local CSV made array task 0 skip, contaminating the first
  attempt with local numbers presented as cluster numbers. Curated,
  publishable results are committed under their own directory (e.g.
  `results/arnes_subset/`) with a `PROVENANCE.md`.
- **Merges always write to a new file.** Never merge into or overwrite
  `results/results.csv` — it is the immutable local RTX 3060 baseline.
  Cluster merges go to `results/results_arnes_subset.csv` and similar.
- **Job receipts via `sacct`.** After every cluster run, record the receipt:
  `sacct -j <jobid> --format=JobID,JobName%20,Elapsed,MaxRSS,State,NodeList`.
  Elapsed/MaxRSS per task go into the run's `PROVENANCE.md` — they are the
  input for sizing `--time` and `--mem` on the larger CC18 array.
- Validation outcome: cluster vs. local agreement is exact (bit-identical)
  for RF/XGBoost/LightGBM/CatBoost and ~1e-4 for TabPFN/TabICL (GPU
  nondeterminism, SM86 vs SM90). See `results/arnes_subset/PROVENANCE.md`.

## Conventions
- Each model module fails soft: exceptions are caught and stored in
  `result["error"]` rather than raised, so one failing (dataset, algorithm,
  fold) combo doesn't crash the full benchmark run.
- `random_state=42` used consistently for reproducibility (data split, CV
  folds, and every model's `random_state`).
- All algorithms use **default hyperparameters** — this is intentional per
  the thesis protocol, not an oversight to "fix".
