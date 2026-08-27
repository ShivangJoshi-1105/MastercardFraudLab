"""
Shared interface for every attack agent.

Why a common interface at all: Pillar 2 needs "algorithms and agents that generate and
simulate attacks at scale" (the brief's wording) — a registry of small, composable agents that
all speak the same schema is what makes it possible to (a) add new fraud types cheaply, (b) feed
every agent's output into the same downstream GAN/feature-engineering/classifier pipeline
without special-casing, and (c) let the Streamlit app list and run any agent generically.

Every agent, tabular or graph, ultimately emits **transactions** in PaySim's schema (amount,
type, orig/dest accounts, balances) plus a few extra bookkeeping columns. Graph agents (mule
chains, rings, fan-in bursts) are not a different data type — they are multiple transactions
that share an `incident_id` and form a topology when you draw an edge per transaction between
`nameOrig` and `nameDest`. Keeping one schema for everything means the tabular WGAN-GP, the
graph GAN, and the feature engineering in `src/defend/features.py` can all consume the exact
same table.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# PaySim's native schema, extended with the bookkeeping columns every agent must fill in.
TRANSACTION_COLUMNS = [
    "step",  # simulated hour, 1..744 (30 days), inherited from PaySim's own clock
    "type",  # CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",  # 1 for every row an attack agent emits
    "attack_type",  # which agent produced this row, e.g. "card_testing_burst"
    "incident_id",  # groups rows belonging to one attack instance (one bust-out, one mule chain, ...)
]


@dataclass
class AgentContext:
    """
    Read-only statistics about the real (legitimate) transaction backbone that agents sample
    from, so injected fraud is scaled/timed realistically instead of using made-up constants.

    Why pass this in rather than hard-coding numbers in each agent: PaySim's amount and step
    distributions are what make the backbone "real" — an attack agent that ignores them and uses
    round hand-picked numbers is exactly the kind of low-fidelity synthetic data the brief
    penalizes. Every agent below draws its amounts/timing from these real distributions.
    """

    rng: np.random.Generator
    real_accounts: np.ndarray  # pool of real nameOrig/nameDest strings to attach fraud to
    amount_quantiles: np.ndarray  # e.g. 100 quantiles of real PAYSIM `amount`, for realistic draws
    max_step: int  # last simulated hour in the backbone (PaySim: 744)
    victim_balances: pd.DataFrame  # sample of real (account, balance) pairs, for agents that
    # need to "take over" a plausible real account (ATO, BEC, romance-scam victim) rather than a
    # balance pulled out of thin air
    used_accounts: set = field(default_factory=set)  # accounts already touched by some agent,
    # so two unrelated attack agents don't collide on the same synthetic account id

    def pick_victim(self) -> tuple[str, float]:
        """Pick a real (account, balance) pair to role-play as an ATO/BEC/romance-scam victim."""
        row = self.victim_balances.sample(1, random_state=int(self.rng.integers(0, 2**31 - 1))).iloc[0]
        return str(row["account"]), float(row["balance"])

    def sample_amount(self, low_q: float = 0.0, high_q: float = 1.0) -> float:
        """Draw a realistic amount from the real backbone's distribution, restricted to a
        quantile band (e.g. low_q=0.0, high_q=0.05 for 'small card-testing probes')."""
        n = len(self.amount_quantiles)
        lo, hi = int(low_q * (n - 1)), max(int(high_q * (n - 1)), 1)
        idx = self.rng.integers(lo, hi + 1)
        return float(self.amount_quantiles[idx])

    def new_account_id(self, prefix: str) -> str:
        """Mint a synthetic account id that can't collide with a real PaySim account or with
        another agent's synthetic accounts in the same run."""
        while True:
            candidate = f"{prefix}{self.rng.integers(10_000_000, 99_999_999)}"
            if candidate not in self.used_accounts:
                self.used_accounts.add(candidate)
                return candidate

    def pick_real_account(self) -> str:
        return str(self.rng.choice(self.real_accounts))


class AttackAgent(ABC):
    """
    Base class for every rule-based fraud simulator.

    Subclasses implement `generate_incident`, which produces the transactions for *one* instance
    of the attack (e.g. one bust-out, one mule chain). `generate` calls it repeatedly to produce
    `n_incidents` labeled, independent attack instances — this separation is what lets the
    Streamlit "Generate Attacks" page ask for "50 more card-testing bursts" without each agent
    re-implementing the same looping/incident-id bookkeeping.
    """

    name: str
    category: str  # matches an ATTACK_TAXONOMY.md category, e.g. "C. Transaction-Level / CNP"
    taxonomy_ref: str  # e.g. "C1" — cross-reference back into the taxonomy doc
    is_graph: bool = False  # True if this agent's incidents form a multi-account topology

    @abstractmethod
    def generate_incident(self, ctx: AgentContext, incident_id: str) -> pd.DataFrame:
        """Return the transactions (PaySim schema + bookkeeping columns) for one attack instance."""
        raise NotImplementedError

    def generate(self, ctx: AgentContext, n_incidents: int) -> pd.DataFrame:
        frames = []
        for i in range(n_incidents):
            incident_id = f"{self.name}_{i}_{ctx.rng.integers(0, 1_000_000)}"
            frame = self.generate_incident(ctx, incident_id)
            frame["attack_type"] = self.name
            frame["incident_id"] = incident_id
            frame["isFraud"] = 1
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=TRANSACTION_COLUMNS)
        return pd.concat(frames, ignore_index=True)[TRANSACTION_COLUMNS]


# --- Registry -----------------------------------------------------------------
# A plain dict registry (not a metaclass/plugin framework) is a deliberate simplicity choice:
# with 10 agents total, anything fancier is solving a problem we don't have.
_REGISTRY: dict[str, type[AttackAgent]] = {}


def register_agent(cls: type[AttackAgent]) -> type[AttackAgent]:
    """Class decorator: `@register_agent` above an AttackAgent subclass makes it discoverable
    by name for the pipeline script and the Streamlit app, without a manual import list to
    maintain in two places."""
    _REGISTRY[cls.name] = cls
    return cls


def get_agent(name: str) -> AttackAgent:
    return _REGISTRY[name]()


def all_agents() -> dict[str, type[AttackAgent]]:
    return dict(_REGISTRY)


def tabular_agents() -> dict[str, type[AttackAgent]]:
    return {k: v for k, v in _REGISTRY.items() if not v.is_graph}


def graph_agents() -> dict[str, type[AttackAgent]]:
    return {k: v for k, v in _REGISTRY.items() if v.is_graph}
