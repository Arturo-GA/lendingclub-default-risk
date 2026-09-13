"""Métricas y gráficos de evaluación con el vocabulario de riesgo de crédito.

Accuracy no aparece a propósito: con una tasa de default de ~15-20 % un modelo
que aprueba a todos tiene 80-85 % de accuracy y cero valor. Lo que se reporta
es capacidad de ordenamiento (AUC, Gini, KS), rendimiento sobre la clase
minoritaria (PR-AUC) y calidad de las probabilidades (Brier, calibración).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config as C

matplotlib.use("Agg")


# --------------------------------------------------------------------------- #
# Métricas escalares
# --------------------------------------------------------------------------- #
def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov–Smirnov: máxima separación entre las CDF de buenos y malos."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def compute_metrics(y_true, y_score) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7))),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


# --------------------------------------------------------------------------- #
# Umbral por costo de negocio
# --------------------------------------------------------------------------- #
def choose_threshold(y_true, y_score, cost_fn: float = C.COST_FALSE_NEGATIVE, cost_fp: float = C.COST_FALSE_POSITIVE) -> dict:
    """Umbral que minimiza el costo esperado (elegido en *validación*, nunca en test).

    Rechazar = predecir default. FN = default aprobado; FP = buen cliente rechazado.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    grid = np.linspace(0.01, 0.99, 197)
    costs = []
    for t in grid:
        pred = y_score >= t
        fn = np.sum((pred == 0) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        costs.append(cost_fn * fn + cost_fp * fp)
    best = int(np.argmin(costs))
    t = float(grid[best])
    pred = y_score >= t
    return {
        "threshold": t,
        "cost": float(costs[best]),
        "approval_rate": float(np.mean(~pred)),
        "default_rate_approved": float(y_true[~pred].mean()) if (~pred).any() else float("nan"),
        "recall_default": float(np.sum(pred & (y_true == 1)) / max(np.sum(y_true == 1), 1)),
        "cost_ratio_fn_fp": cost_fn / cost_fp,
    }


def apply_threshold_report(y_true, y_score, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = np.asarray(y_score) >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "approval_rate": (tn + fn) / len(y_true),
        "default_rate_approved": fn / max(tn + fn, 1),
        "default_rate_baseline": float(y_true.mean()),
        "recall_default": tp / max(tp + fn, 1),
        "precision_default": tp / max(tp + fp, 1),
    }


# --------------------------------------------------------------------------- #
# Tabla de deciles (gains table)
# --------------------------------------------------------------------------- #
def decile_table(y_true, y_score, n: int = 10) -> pd.DataFrame:
    """Ordena por score descendente y resume tasa de default y captura acumulada.

    Es la tabla que un comité de riesgo pide antes que cualquier métrica:
    "si rechazo el 20 % peor, ¿cuántos defaults evito y a cuántos buenos pierdo?"
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    df["decile"] = pd.qcut(df["score"].rank(method="first", ascending=False), n, labels=range(1, n + 1))
    g = df.groupby("decile", observed=True).agg(n=("y", "size"), defaults=("y", "sum"), score_min=("score", "min"), score_max=("score", "max"))
    g["default_rate"] = g["defaults"] / g["n"]
    g["cum_defaults_pct"] = g["defaults"].cumsum() / g["defaults"].sum()
    g["cum_goods_pct"] = (g["n"] - g["defaults"]).cumsum() / (g["n"] - g["defaults"]).sum()
    g["lift"] = g["default_rate"] / df["y"].mean()
    return g.reset_index()


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def plot_roc_pr(results: dict[str, tuple[np.ndarray, np.ndarray]], out: Path) -> None:
    """results: {nombre_modelo: (y_true, y_score)}"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, (y, s) in results.items():
        fpr, tpr, _ = roc_curve(y, s)
        p, r, _ = precision_recall_curve(y, s)
        ax1.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, s):.3f})")
        ax2.plot(r, p, label=f"{name} (PR-AUC {average_precision_score(y, s):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="ROC (test out-of-time)")
    ax2.set(xlabel="Recall (defaults detectados)", ylabel="Precisión", title="Precision–Recall (test)")
    ax1.legend(); ax2.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_calibration(results: dict[str, tuple[np.ndarray, np.ndarray]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (y, s) in results.items():
        frac, mean_pred = calibration_curve(y, s, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfecta")
    ax.set(xlabel="Probabilidad predicha", ylabel="Tasa de default observada", title="Calibración (test)")
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_ks(y_true, y_score, out: Path, title: str = "") -> None:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    k = int(np.argmax(tpr - fpr))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(thr, tpr, label="CDF defaults (TPR)")
    ax.plot(thr, fpr, label="CDF buenos (FPR)")
    ax.vlines(thr[k], fpr[k], tpr[k], color="red", ls="--", label=f"KS = {tpr[k]-fpr[k]:.3f}")
    ax.set(xlim=(0, 1), xlabel="Score", ylabel="Proporción acumulada", title=f"KS {title}")
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_decile_table(tab: pd.DataFrame, out: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(tab["decile"].astype(int), tab["default_rate"] * 100, color="#4C72B0")
    ax.set_xticks(tab["decile"].astype(int))
    ax.axhline(tab["defaults"].sum() / tab["n"].sum() * 100, color="k", ls="--", lw=0.8, label="tasa promedio")
    ax.set(xlabel="Decil de score (1 = mayor riesgo)", ylabel="Tasa de default (%)", title=f"Tasa de default por decil {title}")
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# Estabilidad poblacional (PSI)
# --------------------------------------------------------------------------- #
def psi(expected: pd.Series, actual: pd.Series, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index entre una distribución de referencia y otra.

    Bins por cuantiles de la referencia; los faltantes forman su propio bin.
    < 0.10 estable · 0.10–0.25 vigilar · > 0.25 cambio significativo.
    """
    e, a = pd.Series(expected), pd.Series(actual)
    if pd.api.types.is_numeric_dtype(e) and e.nunique() > bins:
        edges = np.unique(np.nanquantile(e.astype(float), np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        eb = pd.cut(e.astype(float), edges, include_lowest=True).astype("object").where(e.notna(), "missing")
        ab = pd.cut(a.astype(float), edges, include_lowest=True).astype("object").where(a.notna(), "missing")
    else:
        eb = e.astype("object").where(e.notna(), "missing")
        ab = a.astype("object").where(a.notna(), "missing")
    cats = sorted(set(eb.unique()) | set(ab.unique()), key=str)
    pe = eb.value_counts(normalize=True).reindex(cats, fill_value=0) + eps
    pa = ab.value_counts(normalize=True).reindex(cats, fill_value=0) + eps
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_table(X_ref: pd.DataFrame, X_new: pd.DataFrame) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(X_ref[c], X_new[c])} for c in X_ref.columns]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = pd.cut(out["psi"], [-np.inf, C.PSI_WARN, C.PSI_ALERT, np.inf], labels=["estable", "vigilar", "inestable"])
    return out
