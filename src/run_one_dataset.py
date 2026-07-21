"""Požene vse algoritme na enem samem OpenML datasetu (SLURM array job).

Namenjeno gruči Arnes: vsak array task obdela en dataset, tako da se
posamezni dataseti računajo vzporedno. Zagon:

    python -m src.run_one_dataset --index N --ids-file scripts/subset_ids.json

Iz JSON datoteke s seznamom OpenML ID-jev vzame N-ti ID, naloži dataset z
obstoječim load_dataset (isti StratifiedKFold foldi kot lokalno) in zapiše
results/per_dataset/<openml_id>.csv. Če izhodna datoteka že obstaja, izpiše
"already done" in se konča z 0 - ponovni zagon arraya torej preskoči že
narejene datasete.
"""

import argparse
import json
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.data import load_dataset  # noqa: E402
from src.models import REGISTRY  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Zagon vseh algoritmov na enem OpenML datasetu.")
    parser.add_argument("--index", type=int, required=True, help="Indeks ID-ja v datoteki z ID-ji")
    parser.add_argument(
        "--ids-file", default="cc18_ids.json", help="JSON datoteka s seznamom OpenML ID-jev"
    )
    args = parser.parse_args()

    with open(args.ids_file) as f:
        ids = json.load(f)
    openml_id = ids[args.index]

    out_dir = os.path.join(REPO_ROOT, "results", "per_dataset")
    out_path = os.path.join(out_dir, f"{openml_id}.csv")
    if os.path.exists(out_path):
        print("already done")
        return

    cache_dir = os.path.join(REPO_ROOT, "data", "openml_cache")
    dataset = load_dataset(openml_id, n_splits=5, random_state=42, cache_dir=cache_dir)
    X, y = dataset["X"], dataset["y"]
    categorical_cols = dataset["categorical_cols"]
    ds_name = dataset["name"]
    print(
        f"=== {ds_name} (OpenML ID {openml_id}): {X.shape[0]} vzorcev, "
        f"{X.shape[1]} atributov, {len(categorical_cols)} kategoričnih ==="
    )

    rows = []
    for algo_name, run_fn in REGISTRY.items():
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
                "error": result["error"],
            })

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Rezultati shranjeni v {out_path}")


if __name__ == "__main__":
    main()
