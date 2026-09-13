"""Herramientas de selección de variables típicas de riesgo de crédito.

Todas reciben únicamente el conjunto de *entrenamiento*: calcular IV, WoE o
correlaciones con la totalidad de los datos filtra información del test hacia
las decisiones de modelado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def woe_iv_table(x: pd.Series, y: pd.Series, bins: int = 10, eps: float = 0.5) -> tuple[pd.DataFrame, float]:
    """Tabla WoE e Information Value de una variable (numérica → deciles).

    WoE = ln(% buenos / % malos); IV = Σ (%buenos − %malos) · WoE.
    `eps` es una corrección de Laplace para bins sin eventos.
    """
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > bins:
        grouped = pd.qcut(x, q=bins, duplicates="drop")
        grouped = grouped.cat.add_categories("missing").fillna("missing")
    else:
        grouped = x.astype("object").fillna("missing")

    tab = pd.crosstab(grouped, y)
    tab = tab.reindex(columns=[0, 1], fill_value=0)
    tab.columns = ["good", "bad"]
    tab["pct_good"] = (tab["good"] + eps) / (tab["good"].sum() + eps * len(tab))
    tab["pct_bad"] = (tab["bad"] + eps) / (tab["bad"].sum() + eps * len(tab))
    tab["woe"] = np.log(tab["pct_good"] / tab["pct_bad"])
    tab["iv"] = (tab["pct_good"] - tab["pct_bad"]) * tab["woe"]
    tab["bad_rate"] = tab["bad"] / (tab["good"] + tab["bad"])
    return tab, float(tab["iv"].sum())


def iv_summary(X: pd.DataFrame, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    """IV de cada columna, ordenado descendente, con la interpretación clásica."""
    rows = []
    for col in X.columns:
        _, iv = woe_iv_table(X[col], y, bins=bins)
        rows.append({"feature": col, "iv": iv})
    out = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    out["strength"] = pd.cut(
        out["iv"],
        bins=[-np.inf, 0.02, 0.10, 0.30, 0.50, np.inf],
        labels=["inútil", "débil", "medio", "fuerte", "sospechoso"],
    )
    return out


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Asociación entre dos categóricas (0 = independientes, 1 = equivalentes)."""
    ct = pd.crosstab(a, b)
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    r, k = ct.shape
    return float(np.sqrt((chi2 / n) / max(min(k - 1, r - 1), 1)))


def correlated_pairs(X: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Pares de numéricas con |ρ Spearman| ≥ threshold."""
    corr = X.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().reset_index()
    pairs.columns = ["feature_a", "feature_b", "abs_corr"]
    return pairs[pairs["abs_corr"] >= threshold].sort_values("abs_corr", ascending=False).reset_index(drop=True)


def prune_correlated(X: pd.DataFrame, y: pd.Series, threshold: float = 0.8) -> list[str]:
    """De cada par correlacionado elimina la variable con menor IV. Devuelve las eliminadas."""
    ivs = iv_summary(X, y).set_index("feature")["iv"]
    drop: set[str] = set()
    for _, row in correlated_pairs(X, threshold).iterrows():
        a, b = row["feature_a"], row["feature_b"]
        if a in drop or b in drop:
            continue
        drop.add(a if ivs[a] < ivs[b] else b)
    return sorted(drop)


# --------------------------------------------------------------------------- #
# Transformador WoE (scorecard) compatible con scikit-learn
# --------------------------------------------------------------------------- #
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Reemplaza cada variable por su Weight of Evidence aprendido en train.

    Numéricas → bins por cuantiles (aprendidos en `fit`); categóricas → una
    categoría por valor. Los faltantes son su propio bin. Es la base de un
    *scorecard*: una regresión logística sobre WoE es lineal, monótona por
    construcción dentro de cada variable y fácil de auditar.
    """

    def __init__(self, bins: int = 10, eps: float = 0.5, min_bin_frac: float = 0.01):
        self.bins = bins
        self.eps = eps
        self.min_bin_frac = min_bin_frac

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X)
        y = pd.Series(np.asarray(y), index=X.index)
        self.edges_: dict[str, np.ndarray] = {}
        self.maps_: dict[str, dict] = {}
        self.feature_names_in_ = np.asarray(X.columns)
        for col in X.columns:
            x = X[col]
            if pd.api.types.is_numeric_dtype(x) and x.nunique() > self.bins:
                edges = np.unique(np.nanquantile(x.astype(float), np.linspace(0, 1, self.bins + 1)))
                edges[0], edges[-1] = -np.inf, np.inf
                self.edges_[col] = edges
                key = self._bin_numeric(x, edges)
            else:
                key = x.astype("object").where(x.notna(), "missing")
                small = key.value_counts(normalize=True)
                rare = set(small[small < self.min_bin_frac].index)
                key = key.where(~key.isin(rare), "other")
            tab, _ = woe_iv_table(key, y, bins=self.bins, eps=self.eps)
            self.maps_[col] = tab["woe"].to_dict()
        return self

    def _bin_numeric(self, x: pd.Series, edges: np.ndarray) -> pd.Series:
        b = pd.cut(x.astype(float), bins=edges, include_lowest=True).astype("string")
        return b.fillna("missing").astype("object")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        out = pd.DataFrame(index=X.index)
        for col in self.feature_names_in_:
            x = X[col]
            if col in self.edges_:
                key = self._bin_numeric(x, self.edges_[col])
            else:
                key = x.astype("object").where(x.notna(), "missing")
                known = set(self.maps_[col])
                key = key.where(key.isin(known), "other")
            mapped = key.map(self.maps_[col])
            out[col] = pd.to_numeric(mapped, errors="coerce").fillna(0.0).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
