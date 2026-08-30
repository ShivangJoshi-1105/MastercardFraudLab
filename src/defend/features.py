"""
Feature engineering for the defense classifier. Two families of features, each catching a
different class of fraud from the taxonomy:

1. **Balance-consistency features** (`errorBalanceOrig`/`errorBalanceDest`) — PaySim's ledger
   should satisfy `newbalance = oldbalance +/- amount` exactly for legitimate transactions; a
   nonzero residual is a well-known strong fraud signal in this dataset (several published PaySim
   studies lean on it), and it's cheap to compute per-row with no history needed.
2. **Velocity/recency features** (trailing 24h transaction count/sum per account, transactions-
   so-far per account) — this is what actually catches the *behavioral* fingerprints the
   taxonomy describes: card-testing bursts and fan-in mule bursts are, almost by definition, a
   spike in trailing count; ATO/bust-out/BEC fraud is characterized by a large amount hitting an
   account with very few (or zero) prior transactions. No single row-level feature captures
   "this account is brand new and just got drained" - you need the account's own recent history,
   which is exactly what these features encode.

Trailing-window stats are computed with a real time-based rolling window (`step` converted to
hours) rather than a fixed row-count window, so "24 hours" means the same thing regardless of how
bursty or sparse a given account's activity is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_HOURS = 24

FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "amount_to_orig_balance",
    "orig_balance_drained",
    "dest_balance_was_zero",
    "orig_txn_count_so_far",
    "dest_txn_count_so_far",
    "orig_velocity_24h",
    "orig_amount_sum_24h",
    "dest_velocity_24h",
    "dest_amount_sum_24h",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


def _trailing_stats(df: pd.DataFrame, group_col: str, window_hours: int = WINDOW_HOURS):
    """For every row, the count/sum of that account's transactions in the trailing `window_hours`
    (inclusive of the row itself). Uses pandas' native `groupby().rolling()` (implemented in
    Cython) rather than a Python-level loop over groups - with legit-sample sizes in the hundreds
    of thousands, a per-group Python loop measurably matters here.

    `groupby().rolling()` returns its result in *grouped* order (all of group A, then all of
    group B, ...), not the original row order, so a `row_id` column is carried through the same
    groupby machinery (via `.apply(lambda s: s)`, which concatenates in that identical grouped
    order) purely to scatter the rolled values back to their original positions afterward."""
    tmp = df[[group_col, "amount"]].copy()
    tmp["time_idx"] = pd.to_timedelta(df["step"].to_numpy(), unit="h")
    tmp["row_id"] = np.arange(len(df))
    tmp = tmp.sort_values("time_idx")  # rolling() requires a monotonic time index per group

    grouped = tmp.set_index("time_idx").groupby(group_col)
    roll_count = grouped["amount"].rolling(f"{window_hours}h").count().to_numpy()
    roll_sum = grouped["amount"].rolling(f"{window_hours}h").sum().to_numpy()
    row_order = grouped["row_id"].apply(lambda s: s).to_numpy()

    counts = np.empty(len(df), dtype=np.float64)
    sums = np.empty(len(df), dtype=np.float64)
    counts[row_order] = roll_count
    sums[row_order] = roll_sum
    return counts, sums


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.reset_index(drop=True).copy()
    for col in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "step"]:
        df[col] = df[col].astype(float)

    df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["amount_to_orig_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["orig_balance_drained"] = ((df["newbalanceOrig"] <= 0.01) & (df["oldbalanceOrg"] > 0)).astype(int)
    df["dest_balance_was_zero"] = (df["oldbalanceDest"] == 0).astype(int)

    df["orig_txn_count_so_far"] = df.groupby("nameOrig").cumcount()
    df["dest_txn_count_so_far"] = df.groupby("nameDest").cumcount()

    df["orig_velocity_24h"], df["orig_amount_sum_24h"] = _trailing_stats(df, "nameOrig")
    df["dest_velocity_24h"], df["dest_amount_sum_24h"] = _trailing_stats(df, "nameDest")

    type_dummies = pd.get_dummies(df["type"], prefix="type")
    for col in ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        df[col] = type_dummies[col] if col in type_dummies else 0

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].astype(float)
