"""Glavni orkestrator: naloži datasete, požene vse algoritme na skupnih 5
foldih in shrani rezultate + dnevnik predobdelave.

Zagon: python -m src.run_benchmark
"""

import os
import sys

import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.data import load_dataset  # noqa: E402
from src.models import REGISTRY  # noqa: E402

CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def write_preprocessing_log(entries, path):
    lines = ["# Dnevnik predobdelave\n"]
    for e in entries:
        lines.append(f"## {e['algorithm']} × {e['dataset']}")
        lines.append(f"- Predobdelava: {e['preprocessing']}")
        if e["raw_error"]:
            lines.append(f"- Napaka na surovih podatkih: `{e['raw_error']}`")
        if e["error"]:
            lines.append(f"- Napaka: `{e['error']}`")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    config = load_config()
    random_state = config["random_state"]
    n_splits = config["n_splits"]
    cache_dir = os.path.join(REPO_ROOT, config["cache_dir"])
    results_csv = os.path.join(REPO_ROOT, config["output"]["results_csv"])
    log_path = os.path.join(REPO_ROOT, config["output"]["preprocessing_log"])

    rows = []
    log_entries = []

    for ds_cfg in config["datasets"]:
        ds_name = ds_cfg["name"]
        print(f"\n=== Nalagam dataset: {ds_name} (OpenML ID {ds_cfg['id']}) ===")
        dataset = load_dataset(
            ds_cfg["id"], n_splits=n_splits, random_state=random_state, cache_dir=cache_dir
        )
        X, y = dataset["X"], dataset["y"]
        categorical_cols = dataset["categorical_cols"]
        print(
            f"  {X.shape[0]} vzorcev, {X.shape[1]} atributov, "
            f"{len(categorical_cols)} kategoričnih stolpcev"
        )

        for algo_name in config["algorithms"]:
            run_fn = REGISTRY[algo_name]
            logged = False

            for fold_idx, (train_idx, test_idx) in enumerate(dataset["folds"]):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                result = run_fn(X_train, y_train, X_test, y_test, categorical_cols)

                status = "OK" if result["error"] is None else f"ERROR: {result['error']}"
                auc_str = f"{result['roc_auc']:.4f}" if result["roc_auc"] is not None else "n/a"
                print(f"  [{ds_name}] {algo_name} fold {fold_idx}: roc_auc={auc_str}, {status}")

                rows.append({
                    "dataset": ds_name,
                    "algorithm": algo_name,
                    "fold": fold_idx,
                    "roc_auc": result["roc_auc"],
                    "train_time_s": result["train_time_s"],
                    "inference_time_s": result["inference_time_s"],
                })

                if not logged:
                    log_entries.append({
                        "dataset": ds_name,
                        "algorithm": algo_name,
                        "preprocessing": result.get("preprocessing", "n/a"),
                        "raw_error": result.get("raw_error"),
                        "error": result["error"],
                    })
                    logged = True

    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    print(f"\nRezultati shranjeni v {results_csv}")

    write_preprocessing_log(log_entries, log_path)
    print(f"Dnevnik predobdelave shranjen v {log_path}")


if __name__ == "__main__":
    main()
