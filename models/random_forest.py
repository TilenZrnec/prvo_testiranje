"""
Random Forest model za credit-g dataset.

Lastnosti:
- robustna obravnava manjkajočih vrednosti
- obdelava kategoričnih spremenljivk
- enoten interface: run()
- merjenje časa treniranja in inferenc
- izračun ROC-AUC in Accuracy
"""

import time
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, accuracy_score


def _build_preprocessor(X):
    """
    Loči numerične in kategorične stolpce.
    """

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns

    numeric_pipe = SimpleImputer(strategy="median")

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    return preprocessor


def run(X_train, y_train, X_test, y_test):
    """
    Glavna funkcija za evaluacijo Random Forest modela.
    """

    result = {
        "model": "RandomForest",
        "roc_auc": None,
        "accuracy": None,
        "train_time_s": None,
        "inf_time_s": None,
        "error": None
    }

    try:
        start_train = time.time()

        preprocessor = _build_preprocessor(X_train)

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        clf = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        clf.fit(X_train, y_train)

        end_train = time.time()

        start_inf = time.time()
        y_pred = clf.predict(X_test)

        # ROC-AUC zahteva verjetnosti
        y_proba = clf.predict_proba(X_test)[:, 1]

        end_inf = time.time()

        result["train_time_s"] = end_train - start_train
        result["inf_time_s"] = end_inf - start_inf

        result["accuracy"] = accuracy_score(y_test, y_pred)
        result["roc_auc"] = roc_auc_score(y_test, y_proba)

        return result

    except Exception as e:
        result["error"] = str(e)
        return result