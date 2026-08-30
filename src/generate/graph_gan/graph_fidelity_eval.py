"""
Graph-native fidelity metrics - deliberately different from `generate/fidelity_eval.py`'s
column-by-column KS tests, because "does this synthetic mule ring look real" is a *structural*
question a per-column statistic can't answer (a synthetic graph could have a perfectly realistic
amount distribution while having a completely wrong topology - a star instead of a cycle, say).

Four checks, each catching a different way a generated graph could be unrealistic:
1. **Degree distribution** — KS statistic between real and synthetic node-degree sequences.
2. **Clustering coefficient** — average local clustering (how "cliquish" neighborhoods are);
   collusive rings should score meaningfully differently from mule chains here.
3. **Cycle presence rate** — fraction of graphs containing at least one directed cycle; this is
   the single most important structural check for the collusive-ring agent specifically, since a
   ring *is*, definitionally, a cycle.
4. **Discriminative score (graph version)** — reuses the trained `GraphDiscriminator` itself:
   its real-vs-fake accuracy on a held-out split is a direct readout of "can a GNN tell these
   apart," the same interpretation as the tabular discriminative score.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from scipy.stats import ks_2samp

from .models import GraphDiscriminator


def _to_networkx(adj: np.ndarray, mask: np.ndarray, edge_threshold: float = 0.3) -> nx.DiGraph:
    n = int(mask.sum())
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] > edge_threshold:
                g.add_edge(i, j)
    return g


def degree_ks(real_adj: np.ndarray, real_mask: np.ndarray, synth_adj: np.ndarray, synth_mask: np.ndarray) -> float:
    def degrees(adj_batch, mask_batch):
        out = []
        for adj, mask in zip(adj_batch, mask_batch):
            g = _to_networkx(adj, mask)
            out.extend(dict(g.degree()).values())
        return np.array(out) if out else np.array([0])

    real_deg = degrees(real_adj, real_mask)
    synth_deg = degrees(synth_adj, synth_mask)
    return float(ks_2samp(real_deg, synth_deg).statistic)


def avg_clustering(adj_batch: np.ndarray, mask_batch: np.ndarray) -> float:
    coeffs = []
    for adj, mask in zip(adj_batch, mask_batch):
        g = _to_networkx(adj, mask).to_undirected()
        if g.number_of_nodes() > 2:
            coeffs.append(nx.average_clustering(g))
    return float(np.mean(coeffs)) if coeffs else 0.0


def cycle_presence_rate(adj_batch: np.ndarray, mask_batch: np.ndarray) -> float:
    has_cycle = []
    for adj, mask in zip(adj_batch, mask_batch):
        g = _to_networkx(adj, mask)
        has_cycle.append(0 if nx.is_directed_acyclic_graph(g) else 1)
    return float(np.mean(has_cycle)) if has_cycle else 0.0


def graph_discriminative_score(
    critic: GraphDiscriminator,
    real_feats, real_adj, real_mask, real_cond,
    synth_feats, synth_adj, synth_mask, synth_cond,
) -> float:
    """AUC-style separability using the trained critic's own scores: near-0.5 (after mapping
    through a sigmoid-of-score-difference proxy) means the critic can't reliably rank real above
    fake, i.e. high fidelity. We report the fraction of pairs where the critic actually scores
    the real graph higher than the synthetic one - 0.5 = indistinguishable, 1.0 = fully separable."""
    with torch.no_grad():
        real_scores = critic(real_feats, real_adj, real_mask, real_cond).squeeze(-1)
        synth_scores = critic(synth_feats, synth_adj, synth_mask, synth_cond).squeeze(-1)
    n = min(len(real_scores), len(synth_scores))
    wins = (real_scores[:n] > synth_scores[:n]).float().mean().item()
    return float(wins)


def graph_fidelity_report(real_feats, real_adj, real_mask, synth_feats, synth_adj, synth_mask, critic=None, real_cond=None, synth_cond=None) -> dict:
    report = {
        "degree_ks": degree_ks(real_adj.numpy(), real_mask.numpy(), synth_adj.numpy(), synth_mask.numpy()),
        "real_avg_clustering": avg_clustering(real_adj.numpy(), real_mask.numpy()),
        "synth_avg_clustering": avg_clustering(synth_adj.numpy(), synth_mask.numpy()),
        "real_cycle_rate": cycle_presence_rate(real_adj.numpy(), real_mask.numpy()),
        "synth_cycle_rate": cycle_presence_rate(synth_adj.numpy(), synth_mask.numpy()),
    }
    if critic is not None:
        report["discriminative_win_rate"] = graph_discriminative_score(
            critic, real_feats, real_adj, real_mask, real_cond, synth_feats, synth_adj, synth_mask, synth_cond
        )
    return report
