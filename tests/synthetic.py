"""Genera un CSV con el mismo esquema que el de Kaggle para pruebas y smoke tests.

No pretende reproducir la distribución real; solo garantiza que el pipeline
corre de punta a punta sin el archivo de 1.6 GB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lcrisk import config as C  # noqa: E402


def make_synthetic(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2010-01-01", "2016-12-01", freq="MS")
    issue = rng.choice(months, n)
    term = rng.choice([36, 60], n, p=[0.75, 0.25])
    grade_idx = rng.integers(0, 7, n)
    grade = np.array(list("ABCDEFG"))[grade_idx]
    sub_grade = np.array([f"{g}{rng.integers(1, 6)}" for g in grade])
    int_rate = 5 + grade_idx * 3 + rng.normal(0, 1, n)
    fico_low = np.clip(rng.normal(690 - grade_idx * 8, 25, n), 620, 845).round(-1)
    annual_inc = np.exp(rng.normal(11.1, 0.5, n)).round()
    loan_amnt = rng.integers(1, 40, n) * 1000
    dti = np.clip(rng.normal(18, 8, n), 0, 60)

    # Target con señal real en las variables de originación
    logit = -2.6 + 0.12 * (int_rate - 13) + 0.02 * (dti - 18) - 0.01 * (fico_low - 690) + 0.3 * (term == 60)
    default = rng.random(n) < 1 / (1 + np.exp(-logit))
    status = np.where(default, "Charged Off", "Fully Paid")
    # Algunos préstamos vigentes, como en el dataset real
    status[rng.random(n) < 0.15] = "Current"

    df = pd.DataFrame({
        "loan_status": status,
        "issue_d": pd.to_datetime(issue).strftime("%b-%Y"),
        "loan_amnt": loan_amnt,
        "term": np.where(term == 36, " 36 months", " 60 months"),
        "int_rate": int_rate.round(2),
        "installment": (loan_amnt / term * (1 + int_rate / 100)).round(2),
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_length": rng.choice(["< 1 year", "1 year", "3 years", "5 years", "10+ years", None], n),
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n, p=[0.45, 0.42, 0.12, 0.01]),
        "annual_inc": annual_inc,
        "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement", "other", "car"], n),
        "addr_state": rng.choice(["CA", "TX", "NY", "FL", "IL", "PA"], n),
        "application_type": rng.choice(["Individual", "Joint App"], n, p=[0.97, 0.03]),
        "initial_list_status": rng.choice(["w", "f"], n),
        "dti": dti.round(2),
        "earliest_cr_line": pd.to_datetime(rng.choice(pd.date_range("1985-01-01", "2008-01-01", freq="MS"), n)).strftime("%b-%Y"),
        "fico_range_low": fico_low,
        "fico_range_high": fico_low + 4,
        "mths_since_last_delinq": np.where(rng.random(n) < 0.5, rng.integers(1, 120, n), np.nan),
        "revol_util": np.clip(rng.normal(50, 25, n), 0, 120).round(1),
    })
    # Resto de columnas de buró: numéricas con NaN en los años antiguos, como en LC
    for col in C.CREDIT_BUREAU:
        if col not in df:
            vals = rng.integers(0, 30, n).astype(float)
            vals[(pd.to_datetime(df["issue_d"], format="%b-%Y") < "2012-06-01") & (rng.random(n) < 0.8)] = np.nan
            df[col] = vals
    # Columnas con fuga, presentes en el crudo pero que NO deben llegar al modelo
    df["last_pymnt_amnt"] = np.where(default, rng.uniform(0, 500, n), rng.uniform(500, 15000, n)).round(2)
    df["total_pymnt"] = rng.uniform(0, 40000, n).round(2)
    df["recoveries"] = np.where(default, rng.uniform(0, 2000, n), 0.0)
    return df


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/synthetic.csv.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic().to_csv(out, index=False, compression="gzip" if out.suffix == ".gz" else None)
    print(f"Escrito {out}")
