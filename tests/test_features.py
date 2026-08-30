"""Sanity tests for feature engineering - the trailing-window velocity stats are hand-vectorized
(see the docstring in `features.py` explaining why a naive per-group Python loop was replaced),
so it's worth pinning down the exact expected values on a small, hand-checkable example rather
than only trusting it on the full dataset."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defend.features import _trailing_stats, engineer_features


@pytest.fixture
def toy_df():
    return pd.DataFrame(
        {
            "step": [1, 2, 3, 25, 26, 1, 50],
            "type": ["PAYMENT"] * 5 + ["TRANSFER"] * 2,
            "amount": [100, 100, 100, 100, 100, 50, 50],
            "nameOrig": ["A", "A", "A", "A", "A", "B", "B"],
            "oldbalanceOrg": [1000] * 7,
            "newbalanceOrig": [900] * 7,
            "nameDest": ["X"] * 7,
            "oldbalanceDest": [0] * 7,
            "newbalanceDest": [100] * 7,
        }
    )


def test_trailing_stats_24h_window_is_exclusive_left_inclusive_right(toy_df):
    counts, sums = _trailing_stats(toy_df, "nameOrig", window_hours=24)
    # account A: steps 1,2,3,25,26 - a 24h trailing window is (t-24, t]
    assert counts.tolist() == [1, 2, 3, 3, 3, 1, 1]
    assert sums.tolist() == [100, 200, 300, 300, 300, 50, 50]


def test_engineer_features_has_no_nans_and_expected_columns(toy_df):
    engineered = engineer_features(toy_df)
    assert not engineered.isna().any().any()
    for col in ["errorBalanceOrig", "errorBalanceDest", "orig_velocity_24h", "dest_velocity_24h", "type_TRANSFER"]:
        assert col in engineered.columns


def test_trailing_stats_independent_across_accounts(toy_df):
    counts, _ = _trailing_stats(toy_df, "nameOrig", window_hours=24)
    # account B's counts must not be inflated by account A's activity
    b_positions = toy_df.index[toy_df["nameOrig"] == "B"]
    assert all(counts[i] <= 2 for i in b_positions)
