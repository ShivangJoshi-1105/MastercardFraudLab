"""
WGAN-GP training loop for the graph GAN - structurally the same Wasserstein + gradient-penalty
recipe as the tabular GAN (`tabular_gan/train.py`), adapted to graph-shaped batches (node
features + adjacency + mask, instead of one flat feature vector). Kept as a near-mirror of the
tabular trainer deliberately: same stabilization trick, same reason it's needed (a naive BCE
discriminator on tiny graphs collapses just as fast as it does on skewed tabular amounts, only
here it collapses onto "always generate a 3-node chain" instead of one amount bucket).

**Documented risk, per the plan:** graph GAN training is empirically less stable than tabular
GAN training - three interacting outputs (features, mask, adjacency) instead of one give the
generator more ways to find a degenerate shortcut. `graph_fidelity_eval.py` is what tells us,
after training, whether this converged to something worth keeping or whether the fallback
(GNN discriminator as a realism scorer + the rule-based topology sampler standing in for
generation) is the honest thing to ship instead.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import Adam

from .models import GraphDiscriminator, GraphGenerator


@dataclass
class GraphTrainConfig:
    noise_dim: int = 32
    hidden_dim: int = 64
    epochs: int = 300
    batch_size: int = 32
    critic_iters: int = 5
    lr: float = 1e-4
    device: str = "cpu"


def graph_gradient_penalty(critic: GraphDiscriminator, real, fake, cond, device):
    r_feats, r_adj, r_mask = real
    f_feats, f_adj, f_mask = fake
    batch = r_feats.size(0)
    eps = torch.rand(batch, 1, 1, device=device)
    eps_mask = eps.squeeze(-1)

    i_feats = (eps * r_feats + (1 - eps) * f_feats).requires_grad_(True)
    i_adj = (eps * r_adj + (1 - eps) * f_adj).requires_grad_(True)
    i_mask = (eps_mask * r_mask + (1 - eps_mask) * f_mask).requires_grad_(True)

    scores = critic(i_feats, i_adj, i_mask, cond)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=[i_feats, i_adj, i_mask],
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )
    flat = torch.cat([g.reshape(batch, -1) for g in grads], dim=1)
    grad_norm = flat.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


def train_graph_gan(node_feats, adj, mask, labels: list[str], cond_vocab: list[str], config: GraphTrainConfig):
    device = torch.device(config.device)
    max_nodes, node_feat_dim = node_feats.shape[1], node_feats.shape[2]
    cond_idx = {a: i for i, a in enumerate(cond_vocab)}
    cond = torch.zeros(len(labels), len(cond_vocab))
    for i, lab in enumerate(labels):
        cond[i, cond_idx[lab]] = 1.0

    gen = GraphGenerator(config.noise_dim, len(cond_vocab), max_nodes, node_feat_dim, config.hidden_dim).to(device)
    critic = GraphDiscriminator(node_feat_dim, len(cond_vocab), config.hidden_dim).to(device)
    opt_g = Adam(gen.parameters(), lr=config.lr, betas=(0.5, 0.9))
    opt_c = Adam(critic.parameters(), lr=config.lr, betas=(0.5, 0.9))

    n = node_feats.size(0)
    batch_size = min(config.batch_size, n)
    history = {"critic_loss": [], "gen_loss": []}

    for epoch in range(config.epochs):
        perm = torch.randperm(n)
        epoch_c, epoch_g, n_batches = 0.0, 0.0, 0
        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            r_feats, r_adj, r_mask, r_cond = node_feats[idx].to(device), adj[idx].to(device), mask[idx].to(device), cond[idx].to(device)

            for _ in range(config.critic_iters):
                noise = torch.randn(batch_size, config.noise_dim, device=device)
                f_feats, f_adj, f_mask = gen(noise, r_cond)
                c_real = critic(r_feats, r_adj, r_mask, r_cond)
                c_fake = critic(f_feats.detach(), f_adj.detach(), f_mask.detach(), r_cond)
                gp = graph_gradient_penalty(critic, (r_feats, r_adj, r_mask), (f_feats.detach(), f_adj.detach(), f_mask.detach()), r_cond, device)
                c_loss = c_fake.mean() - c_real.mean() + 10.0 * gp
                opt_c.zero_grad()
                c_loss.backward()
                opt_c.step()

            noise = torch.randn(batch_size, config.noise_dim, device=device)
            f_feats, f_adj, f_mask = gen(noise, r_cond)
            g_loss = -critic(f_feats, f_adj, f_mask, r_cond).mean()
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            epoch_c += c_loss.item()
            epoch_g += g_loss.item()
            n_batches += 1

        if n_batches:
            history["critic_loss"].append(epoch_c / n_batches)
            history["gen_loss"].append(epoch_g / n_batches)

    return gen, critic, history


def save_graph_checkpoint(path: str | Path, gen: GraphGenerator, cond_vocab: list[str], config: GraphTrainConfig):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator_state": gen.state_dict(),
            "cond_vocab": cond_vocab,
            "noise_dim": config.noise_dim,
            "hidden_dim": config.hidden_dim,
            "max_nodes": gen.max_nodes,
            "node_feat_dim": gen.node_feat_dim,
        },
        path.with_suffix(".pt"),
    )
