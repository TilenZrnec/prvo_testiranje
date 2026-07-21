"""Združi vse CSV-je iz results/per_dataset/ v en skupni CSV.

Zagon: python scripts/merge_results.py <izhodni.csv>

Izpiše število vrstic in datasetov ter koliko vrstic ima neprazno napako.
"""

import argparse
import glob
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Združevanje per-dataset CSV-jev.")
    parser.add_argument("output_csv", help="Pot do izhodnega skupnega CSV-ja")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(REPO_ROOT, "results", "per_dataset", "*.csv")))
    if not paths:
        print("Ni najdenih CSV-jev v results/per_dataset/")
        return

    merged = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    merged.to_csv(args.output_csv, index=False)

    n_errors = merged["error"].notna().sum() if "error" in merged.columns else 0
    print(f"Združenih {len(paths)} datotek -> {args.output_csv}")
    print(f"Vrstic: {len(merged)}, datasetov: {merged['dataset'].nunique()}")
    print(f"Vrstic z neprazno napako: {n_errors}")


if __name__ == "__main__":
    main()
