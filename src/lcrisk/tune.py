"""Optimización de hiperparámetros con Optuna sobre el conjunto de validación temporal."""
from __future__ import annotations

import logging

import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .pipeline import fit, make_model

log = logging.getLogger(__name__)


def _space(trial: optuna.Trial, model: str) -> dict:
    if model == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        }
    if model == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        }
    if model == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True)}
    if model == "scorecard":
        return {"C": trial.suggest_float("C", 1e-3, 10, log=True), "bins": trial.suggest_int("bins", 5, 20)}
    raise ValueError(model)


def tune(
    model: str,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_valid: pd.DataFrame, y_valid: pd.Series,
    num_cols: list[str], cat_cols: list[str], pos_weight: float,
    n_trials: int = 30, timeout: int | None = None,
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = _space(trial, model)
        pipe = make_model(model, num_cols, cat_cols, pos_weight, params)
        fit(pipe, X_train, y_train, X_valid, y_valid)
        auc = roc_auc_score(y_valid, pipe.predict_proba(X_valid)[:, 1])
        trial.set_user_attr("auc_valid", auc)
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    log.info("Mejor AUC valid %.4f con %s", study.best_value, study.best_params)
    return study
