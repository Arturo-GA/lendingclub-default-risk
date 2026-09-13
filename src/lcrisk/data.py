"""Carga de datos, construcción del target y split temporal."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)


def load_raw(path: Path | str = C.RAW_FILE, nrows: int | None = None, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Lee solo las columnas necesarias del CSV de Kaggle.

    El archivo original tiene 151 columnas y ~2.26 M filas; leer únicamente las
    ~70 que se usan reduce memoria y tiempo de carga de forma drástica.
    `extra_cols` permite cargar columnas fuera de la lista blanca (solo para
    análisis, p. ej. la demostración de fuga de información).
    """
    path = Path(path)
    wanted = set(C.RAW_COLUMNS_TO_LOAD) | set(extra_cols or [])
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Descarga el dataset de Kaggle "
            "(wordsforthewise/lending-club) y colócalo en data/raw/. Ver data/README.md."
        )
    log.info("Leyendo %s", path)
    df = pd.read_csv(
        path,
        usecols=lambda c: c in wanted,
        nrows=nrows,
        low_memory=False,
    )
    missing = set(C.RAW_COLUMNS_TO_LOAD) - set(df.columns)
    if missing:
        log.warning("Columnas esperadas ausentes en el archivo: %s", sorted(missing))
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra a préstamos resueltos y crea `default` (1 = Charged Off)."""
    df = df[df[C.STATUS_COL].isin(C.STATUS_MAP)].copy()
    df[C.TARGET] = df[C.STATUS_COL].map(C.STATUS_MAP).astype("int8")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte issue_d y earliest_cr_line ('Dec-2015') a datetime."""
    df = df.copy()
    for col in (C.DATE_COL, "earliest_cr_line"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%b-%Y", errors="coerce")
    return df


def filter_matured(df: pd.DataFrame, as_of: str = C.AS_OF_DATE, min_issue: str = C.MIN_ISSUE_DATE) -> pd.DataFrame:
    """Conserva préstamos cuyo plazo ya venció a la fecha de corte del extracto.

    Sin este filtro, las cosechas recientes solo contienen préstamos resueltos
    anticipadamente y la tasa de default observada queda sesgada.
    """
    term_months = term_to_int(df["term"])
    valid = df[C.DATE_COL].notna() & term_months.notna()
    # Aritmética vectorizada en meses: (año*12 + mes) + plazo
    issue_m = df[C.DATE_COL].dt.year * 12 + df[C.DATE_COL].dt.month
    maturity_m = issue_m + term_months
    as_of_ts, min_ts = pd.Timestamp(as_of), pd.Timestamp(min_issue)
    as_of_m = as_of_ts.year * 12 + as_of_ts.month
    mask = valid & (maturity_m <= as_of_m) & (df[C.DATE_COL] >= min_ts)
    out = df[mask].copy()
    log.info(
        "Filtro de madurez: %s -> %s filas (%.1f%% conservado)",
        f"{len(df):,}", f"{len(out):,}", 100 * len(out) / max(len(df), 1),
    )
    return out


def term_to_int(s: pd.Series) -> pd.Series:
    """' 36 months' -> 36."""
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def temporal_split(
    df: pd.DataFrame,
    fractions: dict[str, float] = C.SPLIT_FRACTIONS,
    date_col: str = C.DATE_COL,
) -> pd.DataFrame:
    """Asigna 'train' / 'valid' / 'test' según el orden temporal de originación.

    Los préstamos más antiguos entrenan, los más recientes evalúan (out-of-time),
    que es como se valida un modelo de admisión en producción.
    """
    assert abs(sum(fractions.values()) - 1.0) < 1e-9, "Las fracciones deben sumar 1"
    df = df.sort_values(date_col, kind="mergesort").copy()
    n = len(df)
    cut_train = int(n * fractions["train"])
    cut_valid = cut_train + int(n * fractions["valid"])
    split = np.array(["test"] * n, dtype=object)
    split[:cut_train] = "train"
    split[cut_train:cut_valid] = "valid"
    df["split"] = split
    for name, g in df.groupby("split"):
        log.info(
            "%-5s: %s filas | issue_d %s → %s | tasa default %.2f%%",
            name, f"{len(g):,}", g[date_col].min().date(), g[date_col].max().date(),
            100 * g[C.TARGET].mean(),
        )
    return df


def make_dataset(raw_path: Path | str = C.RAW_FILE, nrows: int | None = None, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Pipeline completo de datos: carga → target → fechas → madurez → split."""
    df = load_raw(raw_path, nrows=nrows, extra_cols=extra_cols)
    df = build_target(df)
    df = parse_dates(df)
    df = filter_matured(df)
    df = temporal_split(df)
    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: Path | str = C.PROCESSED_FILE, raw_path: Path | str | None = None) -> None:
    """Guarda el parquet y un meta.json con el resumen (fuente, filas, tasa por split)."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    meta = {
        "raw_path": str(raw_path) if raw_path else None,
        "synthetic": bool(raw_path and "synthetic" in str(raw_path)),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "default_rate": float(df[C.TARGET].mean()),
        "issue_min": str(df[C.DATE_COL].min().date()),
        "issue_max": str(df[C.DATE_COL].max().date()),
        "splits": {
            s: {"n": int(len(g)), "default_rate": float(g[C.TARGET].mean()),
                "issue_min": str(g[C.DATE_COL].min().date()), "issue_max": str(g[C.DATE_COL].max().date())}
            for s, g in df.groupby("split")
        },
    }
    (path.parent / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Guardado %s (%s filas, %s columnas)", path, f"{len(df):,}", df.shape[1])


def load_processed(path: Path | str = C.PROCESSED_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero `python scripts/make_dataset.py`.")
    return pd.read_parquet(path)
