"""
Converts one attack incident's transaction rows (a small set of PaySim-schema rows sharing an
`incident_id`, as emitted by the graph agents in `rule_based_agents/graph_agents.py`) into the
fixed-size tensor representation the graph GAN trains on: node features, a weighted adjacency
matrix, and an existence mask.

Node order is assigned by first appearance (source account of the first transaction is node 0,
etc.) rather than sorted alphabetically - GNNs are meant to be order-invariant by construction
(see the discriminator's pooling in `models.py`), so this ordering choice has no effect on what
the model learns, it just needs to be *some* consistent rule.

**Scoping note on fan-in bursts:** a fan-in incident can have up to ~40 feeder accounts, but we
cap represented graphs at `max_nodes` (12) to keep the dense-adjacency GNN tractable on a laptop
CPU within the contest timeline - a fan-in incident larger than that is truncated to its first
`max_nodes - 1` feeders plus the collector. This still captures the star topology that makes
fan-in bursts detectable; it just caps *how wide* a single training example can be, not whether
the shape is represented at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

MAX_NODES = 12
NODE_FEAT_DIM = 2  # [normalized in-degree, normalized out-degree]


def incident_to_graph(incident_df: pd.DataFrame, max_nodes: int = MAX_NODES):
    accounts: list[str] = []
    seen = set()
    for _, row in incident_df.iterrows():
        for acct in (row["nameOrig"], row["nameDest"]):
            if acct not in seen:
                seen.add(acct)
                accounts.append(acct)
    accounts = accounts[:max_nodes]
    idx = {a: i for i, a in enumerate(accounts)}
    n = len(accounts)

    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    total_amount = incident_df["amount"].sum() or 1.0
    for _, row in incident_df.iterrows():
        if row["nameOrig"] not in idx or row["nameDest"] not in idx:
            continue  # truncated feeder, dropped per the scoping note above
        i, j = idx[row["nameOrig"]], idx[row["nameDest"]]
        adj[i, j] += row["amount"] / total_amount  # normalized edge weight, scale-invariant

    in_deg = (adj > 0).sum(axis=0).astype(np.float32)
    out_deg = (adj > 0).sum(axis=1).astype(np.float32)
    denom = max(n, 1)
    node_feats = np.zeros((max_nodes, NODE_FEAT_DIM), dtype=np.float32)
    node_feats[:n, 0] = in_deg[:n] / denom
    node_feats[:n, 1] = out_deg[:n] / denom
    node_feats = np.tanh(node_feats)  # keep in [-1, 1] like the generator's tanh output

    mask = np.zeros(max_nodes, dtype=np.float32)
    mask[:n] = 1.0

    return node_feats, adj, mask


def incidents_to_tensors(attacks_df: pd.DataFrame, agent_names: list[str], max_nodes: int = MAX_NODES):
    """Builds the (node_feats, adj, mask, cond_label) training tensors for every incident of the
    given graph agent types found in `attacks_df` (the output of `simulate.generate_all_attacks`)."""
    subset = attacks_df[attacks_df["attack_type"].isin(agent_names)]
    node_list, adj_list, mask_list, label_list = [], [], [], []
    for incident_id, incident_df in subset.groupby("incident_id"):
        node_feats, adj, mask = incident_to_graph(incident_df, max_nodes)
        node_list.append(node_feats)
        adj_list.append(adj)
        mask_list.append(mask)
        label_list.append(incident_df["attack_type"].iloc[0])

    return (
        torch.tensor(np.stack(node_list)),
        torch.tensor(np.stack(adj_list)),
        torch.tensor(np.stack(mask_list)),
        label_list,
    )
