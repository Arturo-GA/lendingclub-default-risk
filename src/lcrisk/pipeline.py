"""Pipelines de preprocesamiento + modelo.

El preprocesamiento (imputación, escalado, one-hot) vive *dentro* del Pipeline
de scikit-learn, de modo que se ajusta exclusivamente con el conjunto de
entrenamiento y se aplica tal cual a validación y test.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def _num_linear() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])


def _num_tree_transformers(num_cols: list[str]) -> list[tuple]:
    # XGBoost, LightGBM y RandomForest (sklearn ≥ 1.4) manejan NaN de forma nativa
    # y aprenden hacia qué rama enviar el faltante. Se conservan los NaN y se
    # agrega un indicador explícito por columna, útil porque en este dataset el
    # faltante es informativo (variables de buró incorporadas en 2012).
    return [
        ("num", "passthrough", num_cols),
        ("num_missing", MissingIndicator(features="all", error_on_new=False), num_cols),
    ]


def _cat() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse_output=False)),
    ])


def make_preprocessor(num_cols: list[str], cat_cols: list[str], for_trees: bool) -> ColumnTransformer:
    num_part = _num_tree_transformers(num_cols) if for_trees else [("num", _num_linear(), num_cols)]
    return ColumnTransformer(
        num_part + [("cat", _cat(), cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_model(name: str, num_cols: list[str], cat_cols: list[str], pos_weight: float, params: dict[str, Any] | None = None) -> Pipeline:
    """Devuelve un Pipeline preprocesamiento + clasificador listo para `.fit`.

    `pos_weight` = n_neg / n_pos del train; se usa para compensar el desbalance
    en vez de submuestrear (así el test conserva la distribución real).
    """
    params = params or {}
    if name == "logreg":
        clf = LogisticRegression(
            C=params.get("C", 0.1), class_weight="balanced", max_iter=2000, solver="lbfgs",
        )
        return Pipeline([("prep", make_preprocessor(num_cols, cat_cols, for_trees=False)), ("clf", clf)])

    if name == "scorecard":
        # Regresión logística sobre WoE: el clásico scorecard bancario, totalmente auditable.
        from .selection import WoEEncoder

        clf = LogisticRegression(C=params.get("C", 0.5), class_weight="balanced", max_iter=2000, solver="lbfgs")
        prep = ColumnTransformer(
            [("woe", WoEEncoder(bins=params.get("bins", 10)), num_cols + cat_cols)],
            remainder="drop", verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        return Pipeline([("prep", prep), ("clf", clf)])

    if name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 12),
            min_samples_leaf=params.get("min_samples_leaf", 100),
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=C.RANDOM_STATE,
        )
        return Pipeline([("prep", make_preprocessor(num_cols, cat_cols, for_trees=True)), ("clf", clf)])

    if name == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=params.get("n_estimators", 600),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 5),
            min_child_weight=params.get("min_child_weight", 5),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_lambda=params.get("reg_lambda", 1.0),
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=C.RANDOM_STATE,
        )
        return Pipeline([("prep", make_preprocessor(num_cols, cat_cols, for_trees=True)), ("clf", clf)])

    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=params.get("n_estimators", 800),
            learning_rate=params.get("learning_rate", 0.03),
            num_leaves=params.get("num_leaves", 31),
            min_child_samples=params.get("min_child_samples", 100),
            subsample=params.get("subsample", 0.8),
            subsample_freq=1,
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_lambda=params.get("reg_lambda", 1.0),
            scale_pos_weight=pos_weight,
            n_jobs=-1,
            random_state=C.RANDOM_STATE,
            verbose=-1,
        )
        return Pipeline([("prep", make_preprocessor(num_cols, cat_cols, for_trees=True)), ("clf", clf)])

    raise ValueError(f"Modelo desconocido: {name}")


MODEL_NAMES = ["logreg", "scorecard", "random_forest", "xgboost", "lightgbm"]


def fit(pipe: Pipeline, X_train, y_train, X_valid=None, y_valid=None, early_stopping_rounds: int = 50) -> Pipeline:
    """Ajusta el pipeline; para los boosters usa validación temporal como early stopping.

    El preprocesador se ajusta solo con train y luego transforma valid, así que
    la parada temprana no filtra información de valid al preprocesamiento.
    """
    clf = pipe.named_steps["clf"]
    name = type(clf).__name__
    if X_valid is None or name not in ("XGBClassifier", "LGBMClassifier"):
        return pipe.fit(X_train, y_train)

    prep = pipe.named_steps["prep"]
    Xt = prep.fit_transform(X_train, y_train)
    Xv = prep.transform(X_valid)
    if name == "XGBClassifier":
        clf.set_params(early_stopping_rounds=early_stopping_rounds)
        clf.fit(Xt, y_train, eval_set=[(Xv, y_valid)], verbose=False)
    else:
        import lightgbm as lgb

        # first_metric_only: con scale_pos_weight, el binary_logloss (sin pesos) de
        # validación empeora desde las primeras iteraciones y, si el callback lo
        # vigila, detiene el entrenamiento en ~5 árboles aunque el AUC siga
        # mejorando. Se para únicamente por AUC, igual que en XGBoost.
        clf.fit(Xt, y_train, eval_set=[(Xv, y_valid)], eval_metric="auc",
                callbacks=[lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False)])
    return pipe


def pos_weight_from(y) -> float:
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))
