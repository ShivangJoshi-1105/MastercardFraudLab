"""
The 3 network/graph attack agents (see docs/ATTACK_TAXONOMY.md, category D). These exist because
mule-network fraud is fundamentally a *topology* problem: no single transaction in a layering
chain or a collusive ring looks unusual in isolation — it's only visible as a shape in the graph
of who-pays-whom. That's precisely why Pillar 2 also includes a graph GAN (`src/generate/
graph_gan/`) rather than only a tabular one: a row-independent generative model has no way to
represent "these 6 transactions form a cycle."

Each incident still emits ordinary PaySim-schema rows (same as the tabular agents) so the whole
pipeline downstream keeps working off one table — the graph structure is recovered later by
building a directed multigraph from `nameOrig -> nameDest` edges sharing an `incident_id` (see
`src/generate/graph_gan/train.py` and `src/defend/features.py`).
"""

from __future__ import annotations

import pandas as pd

from .base import AgentContext, AttackAgent, register_agent
from .tabular_agents import _row


@register_agent
class MoneyMuleLayeringChainAgent(AttackAgent):
    """Taxonomy D2 — stolen funds pushed through a chain of freshly-created mule accounts before
    a final cash-out, each hop skimming a small "cut" and firing within a short delay of the
    previous one. Topology: a simple directed path, source -> mule_1 -> mule_2 -> ... -> sink."""

    name = "mule_layering_chain"
    category = "D. Money Laundering / Mule Networks"
    taxonomy_ref = "D2"
    is_graph = True

    def generate_incident(self, ctx: AgentContext, incident_id: str) -> pd.DataFrame:
        n_hops = int(ctx.rng.integers(3, 7))
        amount = ctx.sample_amount(0.7, 0.97) * 3
        step = int(ctx.rng.integers(1, ctx.max_step - n_hops * 3))
        accounts = [ctx.new_account_id("MULE") for _ in range(n_hops + 1)]
        rows = []
        current_amount = amount
        for i in range(n_hops):
            skim = current_amount * (0.01 + 0.03 * ctx.rng.random())  # mule's cut
            next_amount = current_amount - skim
            rows.append(
                _row(step, "TRANSFER", accounts[i], current_amount, 0.0, accounts[i + 1], 0.0, next_amount, next_amount)
            )
            current_amount = next_amount
            step += int(ctx.rng.integers(1, 3))  # each hop clears fast, minimizing exposure time
        return pd.DataFrame(rows)


@register_agent
class CollusiveRingAgent(AttackAgent):
    """Taxonomy D5 — a small cluster of complicit accounts cycles money among themselves
    (A -> B -> C -> A) to keep funds nominally "moving" (a stationary balance is itself a red
    flag) while laundering the original source. Topology: a closed directed cycle, optionally
    traversed more than once."""

    name = "collusive_ring"
    category = "D. Money Laundering / Mule Networks"
    taxonomy_ref = "D5"
    is_graph = True

    def generate_incident(self, ctx: AgentContext, incident_id: str) -> pd.DataFrame:
        ring_size = int(ctx.rng.integers(3, 6))
        n_cycles = int(ctx.rng.integers(1, 3))
        accounts = [ctx.new_account_id("RING") for _ in range(ring_size)]
        amount = ctx.sample_amount(0.5, 0.85) * 2
        step = int(ctx.rng.integers(1, ctx.max_step - ring_size * n_cycles * 2))
        rows = []
        current_amount = amount
        for cycle in range(n_cycles):
            for i in range(ring_size):
                src, dst = accounts[i], accounts[(i + 1) % ring_size]
                skim = current_amount * (0.005 + 0.015 * ctx.rng.random())
                next_amount = current_amount - skim
                rows.append(_row(step, "TRANSFER", src, current_amount, 0.0, dst, 0.0, next_amount, next_amount))
                current_amount = next_amount
                step += int(ctx.rng.integers(2, 6))
        return pd.DataFrame(rows)


@register_agent
class CoordinatedFanInMuleBurstAgent(AttackAgent):
    """Taxonomy D4 — many freshly-opened, superficially unrelated accounts each receive a
    moderate sum and funnel it to one collector account almost simultaneously; a "smash and
    grab" variant of the layering chain optimized for speed over stealth. Topology: a star graph,
    all edges directed into one collector node within a narrow time window."""

    name = "fan_in_mule_burst"
    category = "D. Money Laundering / Mule Networks"
    taxonomy_ref = "D4"
    is_graph = True

    def generate_incident(self, ctx: AgentContext, incident_id: str) -> pd.DataFrame:
        n_feeders = int(ctx.rng.integers(8, 40))
        collector = ctx.new_account_id("COLLECT")
        step0 = int(ctx.rng.integers(1, ctx.max_step - 2))
        rows = []
        collected = 0.0
        for _ in range(n_feeders):
            feeder = ctx.new_account_id("FEED")
            amt = ctx.sample_amount(0.2, 0.5)
            new_collected = collected + amt
            rows.append(_row(step0, "TRANSFER", feeder, amt, 0.0, collector, collected, new_collected, amt))
            collected = new_collected
        return pd.DataFrame(rows)
