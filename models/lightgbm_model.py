"""
LightGBM model za credit-g dataset.

Lastnosti:
- GPU podpora (če je na voljo)
- native handling missing values
- categorical support (LightGBM interno)
- enoten interface: run()
- meritve časa treniranja in inferenc
"""

import time
import numpy as np

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score


def run(X_train, y_train, X_test, y_test):
    """
    Trenira LightGBM model in vrne metrike.
    """

    result = {
        "model": "LightGBM",
        "roc_auc": None,
        "accuracy": None,
        "train_time_s": None,
        "inf_time_s": None,
        "error": None
    }

    try:
        # -------------------------------------------------------
        # Label encoding (LightGBM zahteva numerične label-e)
        # -------------------------------------------------------
        if y_train.dtype == "object":
            y_train_enc = y_train.astype("category").cat.codes
            y_test_enc = y_test.astype("category").cat.codes
        else:
            y_train_enc = y_train
            y_test_enc = y_test

        # -------------------------------------------------------
        # Detekcija kategoričnih stolpcev
        # -------------------------------------------------------
        cat_features = [
            i for i, col in enumerate(X_train.columns)
            if str(X_train[col].dtype) in ["object", "category"]
        ]

        # -------------------------------------------------------
        # Model
        # -------------------------------------------------------
        use_gpu = False

        if use_gpu:
            model = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                device="gpu"
            )
        else:
            model = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42
            )

        # -------------------------------------------------------
        # TRAINING
        # -------------------------------------------------------
        start_train = time.time()

        model.fit(
            X_train,
            y_train_enc,
            categorical_feature=cat_features if len(cat_features) > 0 else "auto"
        )

        end_train = time.time()

        # -------------------------------------------------------
        # INFERENCE
        # -------------------------------------------------------
        start_inf = time.time()

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        end_inf = time.time()

        # -------------------------------------------------------
        # METRICS
        # -------------------------------------------------------
        result["train_time_s"] = end_train - start_train
        result["inf_time_s"] = end_inf - start_inf

        result["accuracy"] = accuracy_score(y_test_enc, y_pred)
        result["roc_auc"] = roc_auc_score(y_test_enc, y_proba)

        return result

    except Exception as e:
        result["error"] = str(e)
        return result