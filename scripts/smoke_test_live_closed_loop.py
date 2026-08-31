"""Exercises the exact code path app/pages/5_Closed_Loop.py's live button runs, outside
Streamlit, to verify it completes and produces sane output before relying on it in the UI."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common
from src.defend.features import get_feature_matrix
from src.closed_loop.feedback import run_closed_loop_iteration

t0 = time.time()
model = common.load_defense_model()
test_df = common.load_parquet("test_set.parquet")
train_sample = pd.read_parquet(common.PROCESSED_DIR / "closed_loop_demo_train_sample.parquet")
_, transformer, cond_vocab, _ = common.load_tabular_gan()

rng = np.random.default_rng()
counter = {"n": 0}


def account_minter(prefix):
    counter["n"] += 1
    return f"{prefix}LIVE{counter['n']}{int(rng.integers(0, 999999))}"


y_prob_test_before = model.predict_proba(get_feature_matrix(test_df))[:, 1]
result = run_closed_loop_iteration(
    train_sample, test_df, model, y_prob_test_before, transformer, cond_vocab,
    account_minter, rng, n_harder_samples=150, gan_epochs=25,
)
elapsed = time.time() - t0

print(f"Elapsed: {elapsed:.1f}s")
print(f"Target attack type: {result['target_attack_type']}")
print(f"Detection before: {result['holdout_detection_rate_before']:.3f}")
print(f"Detection after: {result['holdout_detection_rate_after']:.3f}")
assert 0.0 <= result["holdout_detection_rate_before"] <= 1.0
assert 0.0 <= result["holdout_detection_rate_after"] <= 1.0
print("OK")
