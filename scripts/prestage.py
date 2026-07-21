"""Predpriprava na prijavnem vozlišču Arnes (login node nima GPU-ja).

Prenese OpenML datasete v data/openml_cache ter enkrat fitta TabPFN in
TabICL na majhnem naključnem vzorcu, da se uteži modelov preneseta v
lokalni predpomnilnik - računska vozlišča nato tečejo brez dostopa do
interneta (HF_HUB_OFFLINE=1). Zagon: python scripts/prestage.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.data import load_dataset  # noqa: E402

DATASET_IDS = [31, 37, 38]


def main():
    for openml_id in DATASET_IDS:
        dataset = load_dataset(openml_id, n_splits=5, random_state=42)
        print(f"OpenML {openml_id} ({dataset['name']}): preneseno in predpomnjeno")

    # Majhen naključen binarni problem - dovolj, da se prenesejo uteži modelov.
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 5))
    y = np.tile([0, 1], 20)

    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier()
    clf.fit(X, y)
    clf.predict_proba(X[:4])
    print("TabPFN: uteži prenesene in predpomnjene")

    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device="cpu")
    clf.fit(X, y)
    clf.predict_proba(X[:4])
    print("TabICL: uteži prenesene in predpomnjene")


if __name__ == "__main__":
    main()
