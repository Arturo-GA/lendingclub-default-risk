"""Optimiza hiperparámetros de un modelo con Optuna (validación temporal).

Uso:
    python scripts/tune.py --model lightgbm --trials 40
    python scripts/train.py --params reports/metrics/best_params.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcrisk import config as C  # noqa: E402
from lcrisk.data import load_processed  # noqa: E402
from lcrisk.features import engineer, get_xy  # noqa: E402
from lcrisk.logging_utils import setup_logging  # noqa: E402
from lcrisk.pipeline import pos_weight_from  # noqa: E402
from lcrisk.tune import tune  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost", "logreg", "scorecard"])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--timeout", type=int, default=None, help="segundos")
    p.add_argument("--exclude-pricing", action="store_true")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--out", default=str(C.METRICS_DIR / "best_params.json"))
    args = p.parse_args()

    setup_logging()
    df = engineer(load_processed())
    X, y, num, cat = get_xy(df, exclude_pricing=args.exclude_pricing)
    tr, va = df["split"] == "train", df["split"] == "valid"
    Xtr, ytr = X[tr], y[tr]
    if args.sample_frac:
        idx = Xtr.sample(frac=args.sample_frac, random_state=C.RANDOM_STATE).index
        Xtr, ytr = Xtr.loc[idx], ytr.loc[idx]

    study = tune(args.model, Xtr, ytr, X[va], y[va], num, cat, pos_weight_from(ytr),
                 n_trials=args.trials, timeout=args.timeout)

    out = Path(args.out)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing[args.model] = study.best_params
    out.write_text(json.dumps(existing, indent=2))
    study.trials_dataframe().to_csv(C.METRICS_DIR / f"optuna_trials_{args.model}.csv", index=False)
    print(f"Mejor AUC validación: {study.best_value:.4f}\nParámetros guardados en {out}")


if __name__ == "__main__":
    main()
