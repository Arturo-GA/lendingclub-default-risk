"""Genera el summary plot de SHAP y la tabla de importancia para un modelo entrenado.

Uso:
    python scripts/explain.py --model lightgbm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcrisk import config as C  # noqa: E402
from lcrisk.data import load_processed  # noqa: E402
from lcrisk.explain import plot_shap_summary, shap_importance_table, shap_values_for  # noqa: E402
from lcrisk.features import engineer, get_xy  # noqa: E402
from lcrisk.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost", "random_forest"])
    p.add_argument("--exclude-pricing", action="store_true", help="Debe coincidir con cómo se entrenó")
    p.add_argument("--rows", type=int, default=20_000)
    args = p.parse_args()

    setup_logging()
    pipe = joblib.load(C.MODELS_DIR / f"{args.model}.joblib")
    df = engineer(load_processed())
    X, _, _, _ = get_xy(df, exclude_pricing=args.exclude_pricing)
    X_test = X[df["split"] == "test"]

    values, X_prep = shap_values_for(pipe, X_test, max_rows=args.rows)
    plot_shap_summary(values, X_prep, C.FIGURES_DIR / f"shap_summary_{args.model}.png")
    tab = shap_importance_table(values, X_prep)
    tab.to_csv(C.METRICS_DIR / f"shap_importance_{args.model}.csv", index=False)
    print(tab.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
