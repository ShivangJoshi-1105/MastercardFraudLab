"""Exercises the exact code path the Closed Loop page's new 'train on your generated attacks'
button runs, simulating a user who generated attacks on the Generate Attacks page, to verify
correctness and timing (must stay well under 20s) before relying on it in the UI."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common
from src.generate.rule_based_agents import get_agent, AgentContext
from src.closed_loop.feedback import run_session_iteration

t0 = time.time()

# Simulate what load_demo_backbone() would hand the page, without needing Streamlit's cache.
legit_df = pd.read_parquet(common.PROCESSED_DIR / "demo_legit_sample.parquet")
victim_balances = pd.read_parquet(common.PROCESSED_DIR / "demo_victim_balances.parquet")
import json
with open(common.PROCESSED_DIR / "demo_backbone_meta.json") as f:
    meta = json.load(f)

rng = np.random.default_rng(42)
ctx = AgentContext(
    rng=rng,
    real_accounts=legit_df["nameOrig"].astype(str).unique(),
    amount_quantiles=np.array(meta["amount_quantiles"]),
    max_step=meta["max_step"],
    victim_balances=victim_balances,
)

# Simulate a user clicking "Generate" a few times on different attack types.
agent1 = get_agent("card_testing_burst")
agent2 = get_agent("bec_wire_fraud")
session_pool = pd.concat([agent1.generate(ctx, 5), agent2.generate(ctx, 5)], ignore_index=True)
print(f"Simulated session pool: {len(session_pool)} rows across {session_pool['incident_id'].nunique()} incidents")

model = common.load_defense_model()
background_train = pd.read_parquet(common.PROCESSED_DIR / "closed_loop_demo_train_sample.parquet")

t1 = time.time()
result = run_session_iteration(session_pool, legit_df, background_train, model)
t2 = time.time()

print(f"Setup time: {t1 - t0:.2f}s, iteration time: {t2 - t1:.2f}s, total: {t2 - t0:.2f}s")
print(f"Train incidents: {result['n_train_incidents']}, holdout incidents: {result['n_holdout_incidents']}, holdout rows: {result['n_holdout_rows']}")
print(f"Detection before: {result['detection_before']:.3f} -> after: {result['detection_after']:.3f}")
assert 0.0 <= result["detection_before"] <= 1.0
assert 0.0 <= result["detection_after"] <= 1.0
print("OK")
