"""Pomožne funkcije, skupne vsem modelom."""

from sklearn.metrics import roc_auc_score


def compute_roc_auc(y_true, proba):
    """Izračuna ROC-AUC iz predict_proba izhoda; podpira binarno in več-razredno."""
    n_classes = proba.shape[1]
    if n_classes == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
