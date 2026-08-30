"""Sample synthetic mule/ring/fan-in subgraphs from a trained GraphGenerator and turn them back
into ordinary PaySim-schema transaction rows (same bridge-back pattern as the tabular GAN's
`sample.py`) so the rest of the pipeline never has to know whether a row came from a rule-based
agent or a GAN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import GraphGenerator


def load_graph_generator(path: str | Path):
    path = Path(path)
    ckpt = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=False)
    gen = GraphGenerator(ckpt["noise_dim"], len(ckpt["cond_vocab"]), ckpt["max_nodes"], ckpt["node_feat_dim"], ckpt["hidden_dim"])
    gen.load_state_dict(ckpt["generator_state"])
    gen.eval()
    return gen, ckpt["cond_vocab"], ckpt["noise_dim"]


def sample_graph_attacks(
    path: str | Path,
    attack_type: str,
    n_incidents: int,
    account_minter,
    amount_sampler,
    rng: np.random.Generator,
    mask_threshold: float = 0.5,
    edge_threshold: float = 0.3,
) -> pd.DataFrame:
    gen, cond_vocab, noise_dim = load_graph_generator(path)
    if attack_type not in cond_vocab:
        raise ValueError(f"graph GAN wasn't trained on {attack_type!r}; trained on {cond_vocab}")
    cond_idx = cond_vocab.index(attack_type)

    noise = torch.randn(n_incidents, noise_dim)
    cond = torch.zeros(n_incidents, len(cond_vocab))
    cond[:, cond_idx] = 1.0

    with torch.no_grad():
        node_feats, adj, mask = gen(noise, cond)

    node_feats, adj, mask = node_feats.numpy(), adj.numpy(), mask.numpy()
    rows = []
    for b in range(n_incidents):
        active = mask[b] > mask_threshold
        n_active = int(active.sum())
        if n_active < 2:
            continue  # degenerate sample, skip - a real fallback would resample; fine to drop here
        idx_map = np.where(active)[0]
        accounts = {orig_idx: account_minter("GGAN") for orig_idx in idx_map}
        total_amount = amount_sampler(0.6, 0.9) * 2
        incident_id = f"ggan_{attack_type}_{b}_{rng.integers(0, 1_000_000)}"
        step = int(rng.integers(1, 700))
        for i in idx_map:
            for j in idx_map:
                if i == j or adj[b, i, j] <= edge_threshold:
                    continue
                amt = float(adj[b, i, j] * total_amount)
                if amt <= 0:
                    continue
                rows.append(
                    {
                        "step": step,
                        "type": "TRANSFER",
                        "amount": round(amt, 2),
                        "nameOrig": accounts[i],
                        "oldbalanceOrg": round(amt * 1.2, 2),
                        "newbalanceOrig": round(amt * 0.2, 2),
                        "nameDest": accounts[j],
                        "oldbalanceDest": 0.0,
                        "newbalanceDest": round(amt, 2),
                        "isFraud": 1,
                        "attack_type": attack_type,
                        "incident_id": incident_id,
                    }
                )
                step += 1

    from ..rule_based_agents.base import TRANSACTION_COLUMNS

    if not rows:
        return pd.DataFrame(columns=TRANSACTION_COLUMNS)
    return pd.DataFrame(rows)[TRANSACTION_COLUMNS]
