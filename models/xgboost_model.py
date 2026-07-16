import time
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def run(X_train, y_train, X_test, y_test):

    result = {
        "model": "XGBoost",
        "roc_auc": None,
        "accuracy": None,
        "train_time_s": None,
        "inf_time_s": None,
        "error": None
    }

    try:
        # -------------------------------------------------------
        # LABEL ENCODING
        # -------------------------------------------------------
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # -------------------------------------------------------
        # FORCE COPY (VERY IMPORTANT)
        # -------------------------------------------------------
        X_train = X_train.copy()
        X_test = X_test.copy()

        # -------------------------------------------------------
        # FIND CATEGORICAL COLUMNS
        # -------------------------------------------------------
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
        num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

        # -------------------------------------------------------
        # ENCODING
        # -------------------------------------------------------
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        X_train_cat = encoder.fit_transform(X_train[cat_cols])
        X_test_cat = encoder.transform(X_test[cat_cols])

        X_train_num = X_train[num_cols].to_numpy()
        X_test_num = X_test[num_cols].to_numpy()

        X_train_final = np.hstack([X_train_num, X_train_cat])
        X_test_final = np.hstack([X_test_num, X_test_cat])

        # -------------------------------------------------------
        # MODEL (NO CATEGORICAL SUPPORT MODE!)
        # -------------------------------------------------------
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",   # SAFE MODE
            eval_metric="logloss"
        )

        # -------------------------------------------------------
        # TRAIN
        # -------------------------------------------------------
        start_train = time.time()
        model.fit(X_train_final, y_train_enc)
        end_train = time.time()

        # -------------------------------------------------------
        # PREDICT
        # -------------------------------------------------------
        start_inf = time.time()
        y_pred = model.predict(X_test_final)
        y_proba = model.predict_proba(X_test_final)[:, 1]
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