"""Garantiza que ninguna variable posterior a la originación entre al modelo."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lcrisk import config as C
from lcrisk.data import build_target, parse_dates, filter_matured, temporal_split
from lcrisk.features import LeakageError, assert_no_leakage, engineer, get_xy
from synthetic import make_synthetic


def test_whitelist_and_blacklist_are_disjoint():
    assert not set(C.RAW_FEATURES) & set(C.LEAKAGE_COLUMNS)


def test_classic_leaky_columns_are_blacklisted():
    # Las que más habitualmente inflan el AUC en este dataset
    for col in ["last_pymnt_amnt", "last_fico_range_high", "recoveries", "total_rec_prncp", "total_pymnt"]:
        assert col in C.LEAKAGE_COLUMNS


def test_guard_raises_on_leaky_matrix():
    with pytest.raises(LeakageError):
        assert_no_leakage(["loan_amnt", "last_pymnt_amnt"])


def test_feature_matrix_has_no_leakage():
    df = make_synthetic(3000)
    df = temporal_split(filter_matured(parse_dates(build_target(df))))
    X, y, num, cat = get_xy(engineer(df))
    assert_no_leakage(X.columns)
    assert "last_pymnt_amnt" not in X.columns
    assert set(y.unique()) <= {0, 1}
