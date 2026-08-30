"""
Trains the Pillar 3 defense model: XGBoost on the engineered features from `features.py`,
against the combined real-legit + rule-based-attack + GAN-augmented-attack dataset Pillar 2
produces. Two things worth calling out:

- **Group-aware train/test split.** Splitting fraud rows randomly at the row level would leak:
  multiple rows from the same incident (e.g. a 5-hop mule chain) share obvious structure, so if
  some of an incident's rows land in train and others in test, the model isn't being tested on
  anything genuinely unseen. Legit rows are grouped by account for the same reason - an account's
  own velocity history shouldn't straddle the split. `GroupShuffleSplit` on this composite key
  keeps entire incidents/accounts on one side of the split.
- **`scale_pos_weight`** compensates for the fraud/legit imbalance (rule-based attacks + GAN
  augmentation still produce far fewer rows than the legit sample) so the model isn't just
  rewarded for predicting "legit" every time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from .evaluate import evaluate_predictions
from .features import FEATURE_COLUMNS, engineer_features, get_feature_matrix


def _group_key(df: pd.DataFrame) -> pd.Series:
    return np.where(df["attack_type"] == "legit", "legit_" + df["nameOrig"].astype(str), df["incident_id"].astype(str))


def prepare_dataset(combined_df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    engineered = engineer_features(combined_df)
    engineered["label"] = (engineered["attack_type"] != "legit").astype(int)
    groups = _group_key(engineered)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(engineered, groups=groups))
    return engineered.iloc[train_idx].reset_index(drop=True), engineered.iloc[test_idx].reset_index(drop=True)


def train_xgboost(train_df: pd.DataFrame, seed: int = 42) -> xgb.XGBClassifier:
    X = get_feature_matrix(train_df)
    y = train_df["label"]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def train_and_evaluate(combined_df: pd.DataFrame, seed: int = 42):
    train_df, test_df = prepare_dataset(combined_df, seed=seed)
    model = train_xgboost(train_df, seed=seed)

    X_test = get_feature_matrix(test_df)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(test_df["label"].to_numpy(), y_prob)

    return model, train_df, test_df, y_prob, metrics


def save_model(model: xgb.XGBClassifier, metrics: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "fraud_classifier.json"))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)
