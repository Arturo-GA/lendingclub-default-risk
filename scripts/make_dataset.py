"""Construye data/processed/loans.parquet a partir del CSV crudo de Kaggle.

Uso:
    python scripts/make_dataset.py                 # dataset completo
    python scripts/make_dataset.py --nrows 200000  # prueba rápida
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lcrisk import config as C  # noqa: E402
from lcrisk.data import make_dataset, save_processed  # noqa: E402
from lcrisk.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(C.RAW_FILE))
    parser.add_argument("--out", default=str(C.PROCESSED_FILE))
    parser.add_argument("--nrows", type=int, default=None, help="Leer solo N filas (debug)")
    args = parser.parse_args()

    setup_logging()
    df = make_dataset(args.raw, nrows=args.nrows)
    save_processed(df, args.out, raw_path=args.raw)


if __name__ == "__main__":
    main()
