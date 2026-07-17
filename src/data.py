"""Nalaganje datasetov iz OpenML in priprava skupnih 5-fold CV razdelitev.

Za vsak dataset se folde ustvari samo enkrat (StratifiedKFold, isti random_state),
nato jih ponovno uporabijo vsi algoritmi - s tem so rezultati med algoritmi
primerljivi na identičnih train/test razdelitvah.
"""

import os

import openml
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(openml_id, n_splits=5, random_state=42, cache_dir=None):
    """Prenese/predpomni dataset iz OpenML in pripravi CV folde.

    Vrne slovar: name, X (DataFrame), y (LabelEncoded Series), categorical_cols
    (imena kategoričnih stolpcev po OpenML metapodatkih) in folds (seznam
    (train_idx, test_idx) parov pozicijskih indeksov).
    """
    if cache_dir is None:
        cache_dir = os.path.join(REPO_ROOT, "data", "openml_cache")
    openml.config.cache_directory = cache_dir

    dataset = openml.datasets.get_dataset(openml_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    X = X.reset_index(drop=True)
    y = pd.Series(LabelEncoder().fit_transform(y), name="target")

    categorical_cols = [
        col for col, is_cat in zip(attribute_names, categorical_indicator) if is_cat
    ]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))

    return {
        "name": dataset.name,
        "X": X,
        "y": y,
        "categorical_cols": categorical_cols,
        "folds": folds,
    }
