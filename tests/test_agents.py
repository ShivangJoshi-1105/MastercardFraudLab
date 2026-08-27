"""
Sanity tests for the 10 rule-based attack agents. These are deliberately not "does the fraud
look realistic" tests (that's what fidelity_eval.py + human judgement in the Streamlit app are
for) - they check the cheap-but-easy-to-get-wrong stuff: every agent registers, runs without
crashing, and produces the schema the rest of the pipeline assumes (no NaNs, isFraud always 1,
required columns present). Catching a broken agent here is a lot cheaper than catching it after
hours of GAN training on its output.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generate.rule_based_agents import AgentContext, all_agents, get_agent
from src.generate.rule_based_agents.base import TRANSACTION_COLUMNS


@pytest.fixture
def ctx():
    rng = np.random.default_rng(42)
    real_accounts = np.array([f"C{i}" for i in range(1000, 1050)])
    amount_quantiles = np.sort(rng.exponential(scale=5000, size=200))
    victim_balances = pd.DataFrame(
        {
            "account": real_accounts,
            "balance": rng.exponential(scale=8000, size=len(real_accounts)),
        }
    )
    return AgentContext(
        rng=rng,
        real_accounts=real_accounts,
        amount_quantiles=amount_quantiles,
        max_step=744,
        victim_balances=victim_balances,
    )


def test_all_ten_agents_registered():
    agents = all_agents()
    assert len(agents) == 10, f"expected 10 registered agents, found {len(agents)}: {list(agents)}"


@pytest.mark.parametrize("agent_name", list(all_agents().keys()) if all_agents() else [])
def test_agent_generates_valid_transactions(agent_name, ctx):
    agent = get_agent(agent_name)
    df = agent.generate(ctx, n_incidents=5)

    assert len(df) > 0, f"{agent_name} produced no rows"
    assert list(df.columns) == TRANSACTION_COLUMNS
    assert df["isFraud"].eq(1).all()
    assert df["attack_type"].eq(agent_name).all()
    assert df["incident_id"].nunique() == 5, "expected exactly 5 distinct incidents"
    assert not df[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]].isna().any().any()
    assert (df["amount"] > 0).all(), f"{agent_name} produced a non-positive amount"
    assert df["step"].between(1, ctx.max_step + 10).all()
    assert df["type"].isin(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]).all()
