import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lcrisk import config as C
from lcrisk.data import build_target, filter_matured, parse_dates, temporal_split
from lcrisk.features import emp_length_to_years, engineer
from synthetic import make_synthetic


def _prepared(n=5000):
    return temporal_split(filter_matured(parse_dates(build_target(make_synthetic(n)))))


def test_target_is_default_positive():
    df = build_target(make_synthetic(2000))
    assert set(df[C.STATUS_COL].unique()) <= {"Fully Paid", "Charged Off"}
    assert (df.loc[df[C.STATUS_COL] == "Charged Off", C.TARGET] == 1).all()


def test_maturity_filter_drops_unresolved_vintages():
    df = parse_dates(build_target(make_synthetic(5000)))
    out = filter_matured(df, as_of="2018-12-31")
    sixty = out[out["term"].str.contains("60")]
    assert sixty[C.DATE_COL].max() <= pd.Timestamp("2013-12-01")
    thirty_six = out[out["term"].str.contains("36")]
    assert thirty_six[C.DATE_COL].max() <= pd.Timestamp("2015-12-01")


def test_temporal_split_is_ordered():
    df = _prepared()
    assert df.loc[df.split == "train", C.DATE_COL].max() <= df.loc[df.split == "valid", C.DATE_COL].min()
    assert df.loc[df.split == "valid", C.DATE_COL].max() <= df.loc[df.split == "test", C.DATE_COL].min()
    assert abs(df["split"].value_counts(normalize=True)["train"] - 0.70) < 0.01


def test_emp_length_parsing():
    s = pd.Series(["10+ years", "< 1 year", "3 years", None])
    assert emp_length_to_years(s).tolist()[:3] == [10.0, 0.0, 3.0]


def test_engineer_creates_derived_features():
    df = engineer(_prepared(2000))
    for col in ["fico", "credit_history_months", "loan_to_income", "grade_ord", "sub_grade_ord", "term_months"]:
        assert col in df.columns
    assert df["credit_history_months"].min() > 0
    assert df["sub_grade_ord"].between(1, 35).all()
    assert ((df["sub_grade_ord"] - 1) // 5 + 1 == df["grade_ord"]).all()
