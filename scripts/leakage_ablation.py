"""Cuantifica el efecto de incluir variables posteriores a la originación (fuga de información).

Es una verificación de diseño: muestra por qué la lista negra de `config.LEAKAGE_COLUMNS`
existe y qué pasaría si se ignorara. Entrena el mismo modelo tres veces:
  1. solo con variables de originación (este repositorio)
  2. agregando `last_pymnt_amnt` (monto del último pago)
  3. agregando además `last_fico_range_high` y `total_rec_prncp`

Uso:
    python scripts/leakage_ablation.py [--raw data/raw/accepted_2007_to_2018Q4.csv.gz] [--nrows N]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcrisk import config as C  # noqa: E402
from lcrisk.data import make_dataset  # noqa: E402
from lcrisk.evaluate import compute_metrics  # noqa: E402
from lcrisk.features import engineer, feature_lists  # noqa: E402
from lcrisk.logging_utils import setup_logging  # noqa: E402
from lcrisk.pipeline import fit, make_model, pos_weight_from  # noqa: E402

log = logging.getLogger("leakage")

SCENARIOS = {
    "Solo originación (este repo)": [],
    "+ last_pymnt_amnt": ["last_pymnt_amnt"],
    "+ last_pymnt_amnt, last_fico, total_rec_prncp": ["last_pymnt_amnt", "last_fico_range_high", "total_rec_prncp"],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=str(C.RAW_FILE))
    p.add_argument("--nrows", type=int, default=None)
    p.add_argument("--model", default="lightgbm")
    args = p.parse_args()
    setup_logging()

    leaky = sorted({c for cols in SCENARIOS.values() for c in cols})
    df = engineer(make_dataset(args.raw, nrows=args.nrows, extra_cols=leaky))
    num, cat = feature_lists(df)  # la guarda pasa: `leaky` no está en las listas
    y = df[C.TARGET].astype(int)
    tr, va, te = (df["split"] == s for s in ("train", "valid", "test"))
    pw = pos_weight_from(y[tr])

    rows = []
    for name, extra in SCENARIOS.items():
        extra = [c for c in extra if c in df.columns]
        cols = num + extra + cat
        X = df[cols].copy()
        for c in cat:
            X[c] = X[c].astype("object").fillna("missing")
        pipe = make_model(args.model, num + extra, cat, pw, {"n_estimators": 300})
        fit(pipe, X[tr], y[tr], X[va], y[va])
        m = compute_metrics(y[te], pipe.predict_proba(X[te])[:, 1])
        rows.append({"Escenario": name, "AUC test": round(m["auc"], 4), "Gini": round(m["gini"], 4), "KS": round(m["ks"], 4)})
        log.info("%-48s AUC %.4f", name, m["auc"])

    tab = pd.DataFrame(rows)
    out = C.METRICS_DIR / "leakage_ablation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Ablación de fuga de información\n\n"
        "Mismo modelo y mismo split temporal; solo cambian las columnas disponibles.\n\n"
        + tab.to_markdown(index=False) + "\n", encoding="utf-8",
    )
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
