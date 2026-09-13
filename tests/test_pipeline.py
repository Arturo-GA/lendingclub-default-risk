"""Smoke test: los pipelines entrenan y las métricas son coherentes."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lcrisk.data import build_target, filter_matured, parse_dates, temporal_split
from lcrisk.evaluate import choose_threshold, compute_metrics, decile_table
from lcrisk.features import engineer, get_xy
from lcrisk.pipeline import make_model, pos_weight_from
from synthetic import make_synthetic


def test_logreg_and_lightgbm_train_and_score():
    df = engineer(temporal_split(filter_matured(parse_dates(build_target(make_synthetic(6000))))))
    X, y, num, cat = get_xy(df)
    tr, te = df.split == "train", df.split == "test"
    pw = pos_weight_from(y[tr])
    for name, params in [("logreg", None), ("scorecard", None), ("lightgbm", {"n_estimators": 60})]:
        pipe = make_model(name, num, cat, pw, params).fit(X[tr], y[tr])
        s = pipe.predict_proba(X[te])[:, 1]
        m = compute_metrics(y[te], s)
        assert 0.5 < m["auc"] < 1.0, name          # hay señal pero no es perfecto (sin leakage)
        assert abs(m["gini"] - (2 * m["auc"] - 1)) < 1e-9
        assert 0 < m["ks"] < 1
        dec = decile_table(y[te], s)
        assert len(dec) == 10 and np.isclose(dec["cum_defaults_pct"].iloc[-1], 1.0)
        thr = choose_threshold(y[te], s)
        assert 0 < thr["threshold"] < 1


def test_woe_encoder_and_psi():
    import pandas as pd
    from lcrisk.evaluate import psi
    from lcrisk.selection import WoEEncoder

    df = engineer(temporal_split(filter_matured(parse_dates(build_target(make_synthetic(4000))))))
    X, y, num, cat = get_xy(df)
    tr = df.split == "train"
    enc = WoEEncoder(bins=5).fit(X.loc[tr, ["int_rate", "purpose"]], y[tr])
    out = enc.transform(X[["int_rate", "purpose"]])
    assert out.shape == (len(X), 2) and out.notna().all().all()
    assert psi(X.loc[tr, "int_rate"], X.loc[tr, "int_rate"]) < 1e-6      # misma distribución → 0
    assert psi(pd.Series([1.0] * 500 + [2.0] * 500), pd.Series([1.0] * 900 + [2.0] * 100)) > 0.25


def test_shap_values_are_2d_for_every_tree_model():
    from lcrisk.explain import shap_values_for

    df = engineer(temporal_split(filter_matured(parse_dates(build_target(make_synthetic(3000))))))
    X, y, num, cat = get_xy(df)
    tr = df.split == "train"
    pw = pos_weight_from(y[tr])
    for name, params in [("random_forest", {"n_estimators": 20}), ("lightgbm", {"n_estimators": 30}), ("xgboost", {"n_estimators": 30})]:
        pipe = make_model(name, num, cat, pw, params).fit(X[tr], y[tr])
        vals, Xp = shap_values_for(pipe, X[~tr], max_rows=200)
        assert vals.ndim == 2 and vals.shape == Xp.shape, name


def test_lightgbm_early_stopping_stops_on_auc_not_logloss():
    """Regresión: con scale_pos_weight, el binary_logloss (sin pesos) de validación
    empeora desde las primeras iteraciones. Si el early stopping lo vigila, LightGBM
    se detiene en ~3 árboles aunque el AUC siga mejorando (en el dataset real paró en
    la iteración 4 con AUC test 0.69, frente a 573 y 0.715 al parar solo por AUC)."""
    from lcrisk.pipeline import fit

    df = engineer(temporal_split(filter_matured(parse_dates(build_target(make_synthetic(20_000))))))
    X, y, num, cat = get_xy(df)
    tr, va = df.split == "train", df.split == "valid"
    pipe = make_model("lightgbm", num, cat, pos_weight_from(y[tr]), {"n_estimators": 300, "num_leaves": 7})
    fit(pipe, X[tr], y[tr], X[va], y[va])
    assert pipe.named_steps["clf"].best_iteration_ > 20
