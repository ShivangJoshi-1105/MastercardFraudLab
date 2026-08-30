"""
Orchestrates Pillar 2 end-to-end: load the real PaySim backbone, derive realistic statistics
from it, run all 10 rule-based attack agents at scale against those statistics, and hand back
one unified labeled transaction table. This is the file `scripts/run_pipeline.py` calls first;
everything downstream (the two GANs, the classifier, the closed loop) consumes its output.

Why "derive statistics from the real data" is a whole function and not just `df.sample()`: the
agents don't literally copy real transactions - they synthesize new incidents (new accounts, new
attack instances) but need to draw amounts/victims from the *real* distribution so the injected
fraud isn't obviously fabricated (e.g. suspiciously round numbers). See `AgentContext` in
`rule_based_agents/base.py` for how each agent consumes these.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .rule_based_agents import AgentContext, all_agents

RAW_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "paysim.csv"

DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


@dataclass
class Backbone:
    legit_df: pd.DataFrame  # sample of real, legitimate (isFraud==0) transactions
    native_fraud_df: pd.DataFrame  # PaySim's own built-in fraud rows, kept for a baseline comparison
    ctx: AgentContext
    tabular_attack_types: list[str]
    graph_attack_types: list[str]


def load_backbone(path: Path = RAW_PATH, legit_sample_n: int = 300_000, seed: int = 42) -> Backbone:
    rng = np.random.default_rng(seed)
    df = pd.read_csv(path, dtype=DTYPES)

    legit_full = df[df["isFraud"] == 0]
    native_fraud = df[df["isFraud"] == 1].copy()
    legit_df = legit_full.sample(n=min(legit_sample_n, len(legit_full)), random_state=seed).reset_index(drop=True)

    # Real amount distribution (from the full backbone, not just the sample) drives every
    # agent's `ctx.sample_amount()` draws - this is the main lever for "fidelity to real data".
    amount_quantiles = np.quantile(df["amount"].to_numpy(dtype=np.float64), np.linspace(0, 1, 200))

    # A pool of (account, balance) pairs an agent can "take over" as an ATO/BEC/romance-scam
    # victim - sampled once from legit orig accounts with a nonzero balance.
    victim_pool_src = legit_df[legit_df["oldbalanceOrg"] > 1000][["nameOrig", "oldbalanceOrg"]]
    victim_pool_src = victim_pool_src.sample(n=min(20_000, len(victim_pool_src)), random_state=seed)
    victim_balances = victim_pool_src.rename(columns={"nameOrig": "account", "oldbalanceOrg": "balance"}).reset_index(drop=True)

    real_accounts = legit_df["nameOrig"].astype(str).unique()

    ctx = AgentContext(
        rng=rng,
        real_accounts=real_accounts,
        amount_quantiles=amount_quantiles,
        max_step=int(df["step"].max()),
        victim_balances=victim_balances,
    )

    registry = all_agents()
    tabular_types = [name for name, cls in registry.items() if not cls.is_graph]
    graph_types = [name for name, cls in registry.items() if cls.is_graph]

    return Backbone(legit_df, native_fraud, ctx, tabular_types, graph_types)


def generate_all_attacks(backbone: Backbone, incidents_per_agent: int = 150) -> pd.DataFrame:
    """Runs every registered agent `incidents_per_agent` times and concatenates the results.
    150 incidents/agent x 10 agents gives a few thousand labeled fraud rows straight out of the
    rule-based layer alone - before either GAN adds a single synthetic sample on top."""
    registry = all_agents()
    frames = []
    for name, cls in registry.items():
        agent = cls()
        frames.append(agent.generate(backbone.ctx, incidents_per_agent))
    return pd.concat(frames, ignore_index=True)


def save_labeled_dataset(backbone: Backbone, attacks_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone.legit_df.assign(attack_type="legit", incident_id="legit").to_parquet(out_dir / "legit_transactions.parquet")
    attacks_df.to_parquet(out_dir / "rule_based_attacks.parquet")
    backbone.native_fraud_df.to_parquet(out_dir / "native_paysim_fraud.parquet")
