"""Standalone experiment: does the sparsity penalty fix the graph GAN's degenerate 'everything is
connected' solution seen in the first full pipeline run (degree KS 0.928, synthetic cycle rate
1.00 vs real 0.11)? Runs cheaply against a lightweight synthetic backbone so we don't need to
reload the full PaySim CSV just to iterate on this."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generate.rule_based_agents import AgentContext, all_agents
from src.generate.graph_gan.graph_data import incidents_to_tensors
from src.generate.graph_gan.train import GraphTrainConfig, train_graph_gan
from src.generate.graph_gan.graph_fidelity_eval import graph_fidelity_report
import torch

rng = np.random.default_rng(7)
real_accounts = np.array([f"C{i}" for i in range(1000, 2000)])
amount_quantiles = np.sort(rng.exponential(scale=5000, size=200))
victim_balances = pd.DataFrame({"account": real_accounts, "balance": rng.exponential(scale=8000, size=len(real_accounts))})
ctx = AgentContext(rng=rng, real_accounts=real_accounts, amount_quantiles=amount_quantiles, max_step=744, victim_balances=victim_balances)

registry = all_agents()
graph_types = [name for name, cls in registry.items() if cls.is_graph]
frames = [cls().generate(ctx, 150) for name, cls in registry.items() if cls.is_graph]
graph_df = pd.concat(frames, ignore_index=True)

node_feats, adj, mask, labels = incidents_to_tensors(graph_df, graph_types)
print(f"{len(labels)} graph incidents, types: {graph_types}")

for sparsity_weight in [0.0, 0.05, 0.2, 1.0]:
    config = GraphTrainConfig(epochs=150, batch_size=32, sparsity_weight=sparsity_weight)
    gen, critic, history = train_graph_gan(node_feats, adj, mask, labels, graph_types, config)

    with torch.no_grad():
        noise = torch.randn(len(node_feats), config.noise_dim)
        cond_idx_map = {a: i for i, a in enumerate(graph_types)}
        cond = torch.zeros(len(labels), len(graph_types))
        for i, lab in enumerate(labels):
            cond[i, cond_idx_map[lab]] = 1.0
        synth_feats, synth_adj, synth_mask = gen(noise, cond)

    report = graph_fidelity_report(node_feats, adj, mask, synth_feats, synth_adj, synth_mask, critic=critic, real_cond=cond, synth_cond=cond)
    print(f"sparsity_weight={sparsity_weight}: degree_ks={report['degree_ks']:.3f}, "
          f"real_cycle={report['real_cycle_rate']:.2f}, synth_cycle={report['synth_cycle_rate']:.2f}, "
          f"real_clust={report['real_avg_clustering']:.3f}, synth_clust={report['synth_avg_clustering']:.3f}, "
          f"critic_win_rate={report['discriminative_win_rate']:.3f}")
