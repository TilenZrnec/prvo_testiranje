"""TabPFN v2 (tabpfn paket, privzeti hiperparametri).

Eksperiment: podatke podamo v čim bolj surovi obliki (brez naše predobdelave)
in preverimo, kaj model obravnava nativno. Če surovi vhod sproži napako, to
zabeležimo (raw_error) in šele nato uporabimo minimalni popravek - ne
predobdelujemo tiho.

Ugotovljeno na dejanskih podatkih: TabPFN-jeva interna predobdelava zna
obravnavati pandas 'category' stolpce, a odpove na 'object' stolpcih, ki
vsebujejo prazne/manjkajoče vrednosti (npr. stolpec TBG v datasetu 'sick').
Minimalni popravek: kategorične stolpce ordinalno kodiramo v številske kode,
NaN ohranimo (TabPFN NaN obravnava nativno).
"""

import time

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from src.utils import compute_roc_auc

RAW_PREPROCESSING = "raw (brez predobdelave)"
FALLBACK_PREPROCESSING = (
    "fallback: ordinalno kodiranje kategoričnih stolpcev v številske kode "
    "(NaN ohranjen); TabPFN-jeva interna predobdelava ne zna surovih "
    "string/object kategoričnih stolpcev pretvoriti v float"
)


def _fit_predict(clf, X_train, y_train, X_test):
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    proba = clf.predict_proba(X_test)
    inf_time = time.perf_counter() - t0
    return proba, train_time, inf_time


def run(X_train, y_train, X_test, y_test, categorical_cols):
    from tabpfn import TabPFNClassifier

    result = {
        "model": "TabPFN",
        "roc_auc": None,
        "train_time_s": None,
        "inference_time_s": None,
        "error": None,
        "preprocessing": RAW_PREPROCESSING,
        "raw_error": None,
    }

    try:
        clf = TabPFNClassifier(random_state=42)
        proba, train_time, inf_time = _fit_predict(clf, X_train, y_train, X_test)
    except Exception as e:
        result["raw_error"] = str(e)
        result["preprocessing"] = FALLBACK_PREPROCESSING
        try:
            X_train_fb = X_train.copy()
            X_test_fb = X_test.copy()
            if categorical_cols:
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=np.nan,
                )
                X_train_fb[categorical_cols] = enc.fit_transform(X_train_fb[categorical_cols])
                X_test_fb[categorical_cols] = enc.transform(X_test_fb[categorical_cols])
            X_train_fb = X_train_fb.astype(float)
            X_test_fb = X_test_fb.astype(float)

            clf = TabPFNClassifier(random_state=42)
            proba, train_time, inf_time = _fit_predict(clf, X_train_fb, y_train, X_test_fb)
        except Exception as e2:
            result["error"] = f"raw failed: {result['raw_error']}; fallback failed: {e2}"
            return result

    result["train_time_s"] = train_time
    result["inference_time_s"] = inf_time
    result["roc_auc"] = compute_roc_auc(y_test, proba)
    return result
