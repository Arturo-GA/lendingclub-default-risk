"""Ingeniería de variables y guarda contra fuga de información."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import term_to_int


class LeakageError(RuntimeError):
    """Se lanza si una columna de la lista negra llega a la matriz de features."""


def assert_no_leakage(columns) -> None:
    leaked = sorted(set(columns) & set(C.LEAKAGE_COLUMNS))
    if leaked:
        detail = "\n".join(f"  - {c}: {C.LEAKAGE_COLUMNS[c]}" for c in leaked)
        raise LeakageError(f"Columnas con fuga de información en la matriz de features:\n{detail}")


def emp_length_to_years(s: pd.Series) -> pd.Series:
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3, NaN -> NaN."""
    s = s.astype("string")
    out = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    out = out.where(~s.str.contains("<", na=False), 0)
    return out.astype("float")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Construye variables derivadas a partir de las columnas crudas.

    Todas las variables se calculan con información disponible en la fecha de
    originación (`issue_d`). No se usa nada posterior.
    """
    out = df.copy()

    # Términos del préstamo
    out["term_months"] = term_to_int(out["term"])
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"].replace(0, np.nan)
    out["installment_to_income"] = out["installment"] * 12 / out["annual_inc"].replace(0, np.nan)
    out["log_annual_inc"] = np.log1p(out["annual_inc"].clip(lower=0))

    # Buró
    out["fico"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
    if "earliest_cr_line" in out and np.issubdtype(out["earliest_cr_line"].dtype, np.datetime64):
        delta = out[C.DATE_COL] - out["earliest_cr_line"]
        out["credit_history_months"] = (delta.dt.days / 30.44).round()
    out["revol_util"] = pd.to_numeric(out["revol_util"], errors="coerce")
    out["revol_bal_to_income"] = out["revol_bal"] / out["annual_inc"].replace(0, np.nan)
    out["emp_length_years"] = emp_length_to_years(out["emp_length"])

    # Precio (LendingClub) → ordinal
    grade_map = {g: i for i, g in enumerate("ABCDEFG", start=1)}
    out["grade_ord"] = out["grade"].map(grade_map)
    if "sub_grade" in out:
        # 'A1' → 1, 'A5' → 5, 'B1' → 6, ..., 'G5' → 35
        sg = out["sub_grade"].astype("string")
        letter = sg.str[0].map(grade_map)
        digit = pd.to_numeric(sg.str[1], errors="coerce")
        out["sub_grade_ord"] = ((letter - 1) * 5 + digit).astype("float")

    # Columnas crudas ya reemplazadas por su versión derivada
    out = out.drop(columns=["term", "emp_length", "fico_range_low", "fico_range_high",
                            "earliest_cr_line", "grade", "sub_grade"], errors="ignore")
    return out


NUMERIC_FEATURES = [
    "loan_amnt", "int_rate", "installment", "term_months", "loan_to_income",
    "installment_to_income", "log_annual_inc", "dti", "fico",
    "credit_history_months", "revol_bal", "revol_util", "revol_bal_to_income",
    "emp_length_years", "grade_ord", "sub_grade_ord",
    "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec",
    "pub_rec_bankruptcies", "total_acc", "collections_12_mths_ex_med",
    "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd",
    "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "tax_liens", "tot_cur_bal", "tot_hi_cred_lim", "total_bal_ex_mort",
    "total_bc_limit", "total_il_high_credit_limit", "total_rev_hi_lim",
]

CATEGORICAL_FEATURES = [
    "home_ownership", "verification_status", "purpose", "addr_state",
    "application_type", "initial_list_status",
]

PRICING_FEATURES = ["int_rate", "grade_ord", "sub_grade_ord"]


def feature_lists(df: pd.DataFrame, exclude_pricing: bool = False) -> tuple[list[str], list[str]]:
    """Devuelve (numéricas, categóricas) presentes en `df`."""
    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    if exclude_pricing:
        num = [c for c in num if c not in PRICING_FEATURES]
    assert_no_leakage(num + cat)
    return num, cat


def get_xy(df: pd.DataFrame, exclude_pricing: bool = False) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    num, cat = feature_lists(df, exclude_pricing=exclude_pricing)
    X = df[num + cat].copy()
    for c in cat:
        X[c] = X[c].astype("object").fillna("missing")
    y = df[C.TARGET].astype(int)
    return X, y, num, cat
