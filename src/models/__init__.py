"""Register modelov: ime iz config.yaml -> run() funkcija."""

from . import (
    catboost_model,
    lightgbm_model,
    random_forest,
    tabicl_model,
    tabpfn_model,
    xgboost_model,
)

REGISTRY = {
    "random_forest": random_forest.run,
    "xgboost": xgboost_model.run,
    "lightgbm": lightgbm_model.run,
    "catboost": catboost_model.run,
    "tabpfn": tabpfn_model.run,
    "tabicl": tabicl_model.run,
}
