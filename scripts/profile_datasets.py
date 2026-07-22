"""Izpiše velikosti datasetov iz datoteke z ID-ji, urejene padajoče.

Namen je podatkovno podprta izbira SLURM parametrov --time in --mem: največji
dataseti v zbirki določajo, koliko časa in pomnilnika potrebuje posamezen
array task. Rangira po številu celic (vrstice x atributi).

Privzeto bere metapodatke iz OpenML (hitro, brez prenosa celih datasetov -
potrebuje internet). Z --from-cache namesto tega naloži že predpomnjene
datasete prek load_dataset in izpiše dejanske oblike DataFrameov.

Zagon:

    python scripts/profile_datasets.py --ids-file scripts/cc18_ids.json
    python scripts/profile_datasets.py --ids-file scripts/cc18_ids.json --from-cache
"""

import argparse
import json
import os
import sys

import openml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.data import load_dataset  # noqa: E402

CACHE_DIR = os.path.join(REPO_ROOT, "data", "openml_cache")


def profile_from_metadata(ids):
    """Vrne vrstice (id, ime, vrstice, atributi, kategorični, razredi) iz OpenML metapodatkov."""
    listing = openml.datasets.list_datasets(data_id=ids, output_format="dataframe")
    rows = []
    for openml_id in ids:
        if openml_id not in listing.index:
            rows.append((openml_id, "?", None, None, None, None))
            continue
        meta = listing.loc[openml_id]
        rows.append((
            openml_id,
            meta["name"],
            int(meta["NumberOfInstances"]),
            # NumberOfFeatures šteje tudi ciljno spremenljivko, X ima en stolpec manj.
            int(meta["NumberOfFeatures"]) - 1,
            int(meta["NumberOfSymbolicFeatures"]) - 1,
            int(meta["NumberOfClasses"]),
        ))
    return rows


def profile_from_cache(ids):
    """Vrne iste vrstice, a iz dejansko naloženih (predpomnjenih) datasetov."""
    rows = []
    for openml_id in ids:
        dataset = load_dataset(openml_id, n_splits=5, random_state=42, cache_dir=CACHE_DIR)
        X = dataset["X"]
        rows.append((
            openml_id,
            dataset["name"],
            X.shape[0],
            X.shape[1],
            len(dataset["categorical_cols"]),
            int(dataset["y"].nunique()),
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Velikosti datasetov za dimenzioniranje SLURM virov.")
    parser.add_argument(
        "--ids-file",
        default=os.path.join(REPO_ROOT, "scripts", "cc18_ids.json"),
        help="JSON datoteka s seznamom OpenML ID-jev (privzeto scripts/cc18_ids.json)",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Naloži predpomnjene datasete namesto branja metapodatkov z OpenML",
    )
    parser.add_argument("--top", type=int, default=0, help="Izpiši le N največjih (0 = vse)")
    args = parser.parse_args()

    with open(args.ids_file) as f:
        ids = [int(i) for i in json.load(f)]

    rows = profile_from_cache(ids) if args.from_cache else profile_from_metadata(ids)
    # Rangiranje po celicah; dataseti brez metapodatkov gredo na konec.
    rows.sort(key=lambda r: (r[2] or 0) * (r[3] or 0), reverse=True)
    if args.top:
        rows = rows[: args.top]

    print(f"{'id':>6}  {'ime':<26} {'vrstice':>9} {'atributi':>9} {'kateg.':>7} {'razredi':>8} {'celice':>12}")
    total_cells = 0
    for openml_id, name, n_rows, n_feat, n_cat, n_cls in rows:
        cells = (n_rows or 0) * (n_feat or 0)
        total_cells += cells
        print(
            f"{openml_id:>6}  {name[:26]:<26} {n_rows:>9} {n_feat:>9} {n_cat:>7} {n_cls:>8} {cells:>12,}"
        )
    print(f"\nSkupaj {len(rows)} datasetov, {total_cells:,} celic")


if __name__ == "__main__":
    main()
