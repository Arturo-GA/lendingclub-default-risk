"""Explicabilidad con SHAP para el modelo de árboles seleccionado."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")


def shap_values_for(pipe: Pipeline, X: pd.DataFrame, max_rows: int = 20_000, seed: int = 42) -> tuple[np.ndarray, pd.DataFrame]:
    """Calcula SHAP sobre una muestra del conjunto ya preprocesado."""
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    X_prep = pipe.named_steps["prep"].transform(X)
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    values = explainer.shap_values(X_prep)
    # RandomForest devuelve una lista [clase0, clase1] o un arreglo (n, features, clases);
    # XGBoost/LightGBM devuelven directamente (n, features) para la clase positiva.
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values, X_prep


def plot_shap_summary(values: np.ndarray, X_prep: pd.DataFrame, out: Path, max_display: int = 25) -> None:
    plt.figure()
    shap.summary_plot(values, X_prep, max_display=max_display, show=False, plot_size=(11, 7))
    plt.title("Contribución de cada variable a la probabilidad de default (SHAP)")
    plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()


def shap_importance_table(values: np.ndarray, X_prep: pd.DataFrame) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": X_prep.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
