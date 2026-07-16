"""
Nalaganje OpenML dataseta credit-g.

Ta modul:
- prenese dataset iz OpenML,
- ga shrani v lokalni cache,
- izpiše osnovne informacije,
- pripravi train/test razdelitev,
- pripravi 5-fold StratifiedKFold objekt.
"""

import os

import openml
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)


CACHE_DIR = os.path.join(
    "/mnt/d/fri/Diplomska/PRVO TESTIRANJE",
    "data"
)

openml.config.cache_directory = CACHE_DIR


def load_dataset(random_state: int = 42):
    """
    Vrne:
        X_train
        X_test
        y_train
        y_test
        cv
    """

    print("Prenašam dataset credit-g iz OpenML...")

    dataset = openml.datasets.get_dataset(31)

    X, y, categorical, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe"
    )

    print("\n===== DATASET =====")
    print(f"Vzorcev: {len(X)}")
    print(f"Atributov: {X.shape[1]}")
    print()

    print("Razredi:")

    print(y.value_counts())

    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        cv
    )


if __name__ == "__main__":

    load_dataset()