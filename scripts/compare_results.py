"""Primerjava lokalnih in Arnes rezultatov (sanity check prenosa na gručo).

Zagon: python scripts/compare_results.py <lokalni.csv> <arnes.csv>

Združi oba CSV-ja po (dataset, algorithm, fold) in izpiše povprečno ter
največjo absolutno razliko ROC-AUC po algoritmih in 5 najslabših vrstic.
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Primerjava dveh CSV-jev z rezultati.")
    parser.add_argument("local_csv", help="CSV z lokalnimi rezultati")
    parser.add_argument("arnes_csv", help="CSV z rezultati z Arnesa")
    args = parser.parse_args()

    local = pd.read_csv(args.local_csv)
    arnes = pd.read_csv(args.arnes_csv)

    merged = local.merge(
        arnes, on=["dataset", "algorithm", "fold"], suffixes=("_local", "_arnes")
    )
    if len(merged) == 0:
        print(
            "OPOZORILO: združevanje ni dalo nobene vrstice - imena datasetov se "
            "med datotekama morda razlikujejo."
        )
        return

    merged["abs_delta"] = (merged["roc_auc_local"] - merged["roc_auc_arnes"]).abs()

    print(f"Skupnih vrstic: {len(merged)}\n")
    print("|delta roc_auc| po algoritmih:")
    print(merged.groupby("algorithm")["abs_delta"].agg(["mean", "max"]).to_string())

    worst = merged.nlargest(5, "abs_delta")
    print("\n5 najslabših vrstic:")
    cols = ["dataset", "algorithm", "fold", "roc_auc_local", "roc_auc_arnes", "abs_delta"]
    print(worst[cols].to_string(index=False))


if __name__ == "__main__":
    main()
