"""
The 7 behavioral/tabular attack agents (see docs/ATTACK_TAXONOMY.md for the fraud rationale
behind each). Each agent's job is to encode the *exact transactional fingerprint* a real
GenAI-enabled attack would leave — not to simulate the GenAI step itself (no model here writes a
phishing email or clones a voice; that's covered narratively in `llm_strategist.py`). Getting
these fingerprints right matters more than it might look: this is the ground-truth logic the
tabular WGAN-GP learns to imitate at scale, so a sloppy agent here produces a sloppy GAN.

All balance bookkeeping is done by hand (not by PaySim) because these are entirely new,
synthetic incidents grafted onto the real backbone — we control the whole ledger for the
accounts involved, so we can (and should) keep debits/credits internally consistent, which is a
basic realism bar synthetic fraud data has to clear.
"""

from __future__ import annotations

import pandas as pd

from .base import AgentContext, AttackAgent, register_agent


def _row(step, ttype, orig, orig_old, orig_new, dest, dest_old, dest_new, amount):
    return {
        "step": step,
        "type": ttype,
        "amount": round(amount, 2),
        "nameOrig": orig,
        "oldbalanceOrg": round(orig_old, 2),
        "newbalanceOrig": round(orig_new, 2),
        "nameDest": dest,
        "oldbalanceDest": round(dest_old, 2),
        "newbalanceDest": round(dest_new, 2),
    }


@register_agent
class CardTestingBurstAgent(AttackAgent):
    """Taxonomy C1/B3 — a burst of many small, rapid probing transactions from one compromised
    account against many distinct merchants, in a tight time window. Real card-testing bots
    (and the credential-stuffing bots of B3, which share this exact shape) succeed by hiding a
    few live cards/credentials inside a flood of attempts, so volume + tight timing + low value
    is the fingerprint, not any single transaction's size."""

    name = "card_testing_burst"
    category = "C. Transaction-Level / Card-Not-Present"
    taxonomy_ref = "C1"

    def generate_incident(self, ctx: AgentContext, incident_id: str) -> pd.DataFrame:
        orig = ctx.new_account_id("CT")
        balance = ctx.sample_amount(0.5, 0.9) * 3  # compromised account has some real funds
        step0 = int(ctx.rng.integers(1, ctx.max_step - 2))
        n_probes = int(ctx.rng.integers(8, 30))
        rows = []
        running = balance
        for i in range(n_probes):
            amt = ctx.sample_amount(0.0, 0.03)  # very small probing amounts
            dest = ctx.new_account_id("M")  # PaySim merchant-style prefix
            new_running = max(running - amt, 0)
            rows.append(
                _row(
                    step0 + i // 6,  # several probes land in the same simulated hour
                    "PAYMENT",
                    orig,
                    running,
                    new_running,
                    dest,
                    0.0,
                    amt,
                    amt,
                )
            )
            running = new_running
        return pd.DataFrame(rows)


@register_agent
class AdaptiveStructuringAgent(AttackAgent):
    """Taxonomy C3 — one large amount deliberately split into many transfers that stay just
    under a reporting threshold, timed to look like ordinary small transfers rather than one
    obvious lump. The "adaptive"/GenAI angle is in the *sizing strategy* (hug the threshold,
    jitter amounts so they don't look round) — a naive attacker would just split evenly, which
    is far easier to flag than threshold-hugging with jitter."""

    name = "adaptive_structuring"
    category = "C. Transaction-Level / Card-Not-Present"
    taxonomy_ref = "C3"
    REPORTING_THRESHOLD = 200_000.0

    def generate_incident(self, ctx: AgentContext, incident_id: str) -> pd.DataFrame:
        orig = ctx.new_account_id("ST")
        dest = ctx.new_account_id("ST")
        total = ctx.sample_amount(0.85, 0.99) * 4  # a genuinely large sum to launder
        step0 = int(ctx.rng.integers(1, ctx.max_step - 48))
        rows = []
        remaining = total
        orig_balance = total
        dest_balance = 0.0
        step = step0
        while remaining > 0:
            frac = 0.80 + 0.15 * ctx.rng.random()  # hug 80-95% of the threshold
            amt = min(remaining, self.REPORTING_THRESHOLD * frac)
            new_orig_balance = max(orig_balance - amt, 0)
            new_dest_balance = dest_balance + amt
            rows.append(
                _row(step, "TRANSFER", orig, orig_balance, new_orig_balance, dest, dest_balance, new_dest_balance, amt)
            )
            orig_balance, dest_balance = new_orig_balance, new_dest_balance
            remaining -= amt
            step += int(ctx.rng.integers(2, 9))  # spread across hours/days, not all at once
        return pd.DataFrame(rows)


@register_agent
class AccountTakeoverVelocitySpikeAgent(AttackAgent):
    """Taxonomy B1/E3 — a dormant, previously well-behaved account is drained almost instantly.
    The defining fingerprint (per the taxonomy doc) is that this looks like a *clean* auth event
    — a voice clone or smishing script passed verification rather than brute-forcing it — so
    there's no failed-attempt trail, just a step-change from dormant to fully drained."""

    name = "ato_velocity_spike"
    category = "B. Account Takeover / Authentication Bypass"
    taxonomy_ref = "B1"

    def generate_incident(self, ctx: AgentContext, incident_id: str) -> pd.DataFrame:
        victim, balance = ctx.pick_victim()
        dest = ctx.new_account_id("ATO")
        step = int(ctx.rng.integers(1, ctx.max_step))
        drain_frac = 0.90 + 0.10 * ctx.rng.random()
        amt = balance * drain_frac
        rows = [_row(step, "CASH_OUT", victim, balance, balance - amt, dest, 0.0, amt, amt)]
        # a second, smaller mop-up transaction a few hours later is common once the first clears
        if ctx.rng.random() < 0.4:
            amt2 = (balance - amt) * (0.5 + 0.5 * ctx.rng.random())
            dest2 = ctx.new_account_id("ATO")
            rows.append(
                _row(step + int(ctx.rng.integers(1, 4)), "TRANSFER", victim, balance - amt, balance - amt - amt2, dest2, 0.0, amt2, amt2)
            )
        return pd.DataFrame(rows)


@register_agent
class SyntheticIdentityBustOutAgent(AttackAgent):
    """Taxonomy A3/A4 — a fabricated identity opens an account, behaves like a normal customer
    for a while to build trust/limit, then drains everything at once and goes dark. The
    "seasoning" period is the point: without it, this would look identical to a simple ATO
    spike, but the weeks of mundane small transactions beforehand are exactly what lets a
    synthetic identity pass velocity-based new-account risk rules."""

    name = "synthetic_identity_bustout"
    category = "A. Identity & Onboarding Fraud"
    taxonomy_ref = "A3"

    def generate_incident(self, ctx: AgentContext, incident_id: str) -> pd.DataFrame:
        acct = ctx.new_account_id("SYN")
        step0 = int(ctx.rng.integers(1, ctx.max_step - 200))
        n_seasoning = int(ctx.rng.integers(6, 16))
        rows = []
        balance = 0.0
        step = step0
        for _ in range(n_seasoning):
            amt_in = ctx.sample_amount(0.1, 0.4)
            new_balance = balance + amt_in
            rows.append(_row(step, "CASH_IN", acct, balance, new_balance, ctx.new_account_id("M"), 0.0, 0.0, amt_in))
            balance = new_balance
            amt_out = amt_in * (0.3 + 0.4 * ctx.rng.random())
            new_balance2 = max(balance - amt_out, 0)
            rows.append(_row(step + 1, "PAYMENT", acct, balance, new_balance2, ctx.new_account_id("M"), 0.0, amt_out, amt_out))
            balance = new_balance2
            step += int(ctx.rng.integers(8, 30))
        # bust-out: max the account out and disappear
        bust_amt = balance + ctx.sample_amount(0.6, 0.95) * 3
        dest = ctx.new_account_id("BUST")
        rows.append(_row(step, "CASH_OUT", acct, balance, 0.0, dest, 0.0, bust_amt, bust_amt))
        return pd.DataFrame(rows)


@register_agent
class BecWireFraudProxyAgent(AttackAgent):
    """Taxonomy E1 — a deepfake-executive-voice or BEC email convinces an authorized employee to
    wire funds themselves, so this bypasses ATO defenses entirely (no account was "taken over").
    Fingerprint: one large, urgent, first-time-payee wire, often off-hours, from an account with
    otherwise unremarkable history."""

    name = "bec_wire_fraud"
    category = "E. Social Engineering / Authorized Push Payment"
    taxonomy_ref = "E1"

    def generate_incident(self, ctx: AgentContext, incident_id: str) -> pd.DataFrame:
        victim, balance = ctx.pick_victim()
        dest = ctx.new_account_id("BEC")
        day = int(ctx.rng.integers(0, max(ctx.max_step // 24 - 1, 1)))
        off_hour = int(ctx.rng.choice([22, 23, 0, 1, 2, 5, 6]))
        step = min(day * 24 + off_hour, ctx.max_step - 1)
        amt = balance * (0.4 + 0.5 * ctx.rng.random())
        return pd.DataFrame([_row(step, "TRANSFER", victim, balance, balance - amt, dest, 0.0, amt, amt)])


@register_agent
class RomanceScamProxyAgent(AttackAgent):
    """Taxonomy E2 — an LLM-run romance/pig-butchering scam guides a victim into self-authorizing
    an escalating series of transfers to one new payee over time. Every individual transaction
    looks like an ordinary transfer; only the escalating pattern to a single new counterparty
    over the whole incident gives it away, which is what makes this one of the hardest agents
    for the defense model to catch."""

    name = "romance_scam_proxy"
    category = "E. Social Engineering / Authorized Push Payment"
    taxonomy_ref = "E2"

    def generate_incident(self, ctx: AgentContext, incident_id: str) -> pd.DataFrame:
        victim, balance = ctx.pick_victim()
        dest = ctx.new_account_id("ROM")
        n_transfers = int(ctx.rng.integers(3, 7))
        step = int(ctx.rng.integers(1, ctx.max_step - n_transfers * 20))
        rows = []
        remaining_balance = balance
        base_amt = ctx.sample_amount(0.2, 0.4)
        for i in range(n_transfers):
            amt = min(base_amt * (1.4 ** i), remaining_balance * 0.9)
            new_balance = remaining_balance - amt
            rows.append(_row(step, "TRANSFER", victim, remaining_balance, new_balance, dest, 0.0, amt, amt))
            remaining_balance = new_balance
            step += int(ctx.rng.integers(10, 30))
            if remaining_balance <= 0:
                break
        return pd.DataFrame(rows)


@register_agent
class ClassifierEvasionProbeAgent(AttackAgent):
    """Taxonomy C2 — an adversary (or our own red-team loop, see src/closed_loop/feedback.py)
    with query access to a fraud classifier iteratively perturbs a transaction to find the
    smallest change that flips its prediction from fraud to legitimate. Without a live model to
    query yet, this standalone version searches near a *known* structuring threshold as a
    reasonable stand-in; `set_scorer()` lets the closed loop swap in real hill-climbing against
    the actual trained classifier once one exists (see Day 4 of the build)."""

    name = "classifier_evasion_probe"
    category = "C. Transaction-Level / Card-Not-Present"
    taxonomy_ref = "C2"
    REPORTING_THRESHOLD = 200_000.0

    def __init__(self):
        self._scorer = None  # optional callable: row_dict -> fraud probability

    def set_scorer(self, scorer):
        self._scorer = scorer

    def generate_incident(self, ctx: AgentContext, incident_id: str) -> pd.DataFrame:
        orig = ctx.new_account_id("EV")
        dest = ctx.new_account_id("EV")
        step = int(ctx.rng.integers(1, ctx.max_step))
        # Start near the known threshold and jitter closer to it over a few probes -
        # a simple hill-climb even without a live scorer plugged in.
        amt = self.REPORTING_THRESHOLD * (0.9 + 0.09 * ctx.rng.random())
        balance = amt * 1.5
        rows = [_row(step, "TRANSFER", orig, balance, balance - amt, dest, 0.0, amt, amt)]
        for i in range(int(ctx.rng.integers(1, 4))):
            amt = amt * (0.95 + 0.03 * ctx.rng.random())  # nudge down, staying near the edge
            step += 1
            new_balance = max(balance - amt, 0)
            rows.append(_row(step, "TRANSFER", orig, balance, new_balance, dest, 0.0, amt, amt))
            balance = new_balance
        return pd.DataFrame(rows)
