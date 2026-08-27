"""
Training loop for the tabular GAN. Supports two loss modes via `loss_type`:

- `"vanilla"` — classic BCE discriminator loss. Kept in specifically as a teaching/demo artifact:
  running this on skewed transaction amounts visibly mode-collapses within a few dozen epochs
  (watch the generator loss flatten while the critic loss keeps sinking - the discriminator
  wins and stops giving useful gradient). Run `scripts/run_pipeline.py --demo-mode-collapse` to
  reproduce this before looking at the fixed version.
- `"wgan_gp"` (default, what the pipeline actually uses) — Wasserstein loss with gradient
  penalty, per `models.py`'s docstring. This is what trains stably.

The generator is conditioned on attack_type: `cond_vocab` fixes the ordering of the one-hot
condition vector, so sampling can request "give me N more of attack type k" (see `sample.py`).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from .data_transformer import TabularDataTransformer
from .models import Critic, Generator, apply_activations, gradient_penalty


@dataclass
class TrainConfig:
    noise_dim: int = 64
    hidden_dim: int = 128
    n_blocks: int = 3
    epochs: int = 150
    batch_size: int = 256
    critic_iters: int = 5
    lr: float = 2e-4
    loss_type: str = "wgan_gp"  # or "vanilla"
    device: str = "cpu"


def train_tabular_gan(real_df: pd.DataFrame, cond_vocab: list[str], transformer: TabularDataTransformer, config: TrainConfig):
    device = torch.device(config.device)
    X = transformer.transform(real_df)
    cond_idx = {a: i for i, a in enumerate(cond_vocab)}
    labels = real_df["attack_type"].map(cond_idx).to_numpy()
    C = np.zeros((len(real_df), len(cond_vocab)), dtype=np.float32)
    C[np.arange(len(real_df)), labels] = 1.0

    dataset = TensorDataset(torch.tensor(X), torch.tensor(C))
    batch_size = min(config.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    gen = Generator(config.noise_dim, len(cond_vocab), config.hidden_dim, transformer.output_dim, config.n_blocks).to(device)
    critic = Critic(transformer.output_dim, len(cond_vocab), config.hidden_dim).to(device)
    opt_g = Adam(gen.parameters(), lr=config.lr, betas=(0.5, 0.9))
    opt_c = Adam(critic.parameters(), lr=config.lr, betas=(0.5, 0.9))
    spec = transformer.activation_spec()
    bce = nn.BCEWithLogitsLoss()

    history = {"critic_loss": [], "gen_loss": []}
    critic_iters = config.critic_iters if config.loss_type == "wgan_gp" else 1

    for epoch in range(config.epochs):
        epoch_c_loss, epoch_g_loss, n_batches = 0.0, 0.0, 0
        for real_x, real_c in loader:
            real_x, real_c = real_x.to(device), real_c.to(device)

            for _ in range(critic_iters):
                noise = torch.randn(real_x.size(0), config.noise_dim, device=device)
                fake_x = apply_activations(gen(noise, real_c), spec)
                c_real = critic(real_x, real_c)
                c_fake = critic(fake_x.detach(), real_c)
                if config.loss_type == "wgan_gp":
                    gp = gradient_penalty(critic, real_x, fake_x.detach(), real_c, device)
                    c_loss = c_fake.mean() - c_real.mean() + 10.0 * gp
                else:
                    c_loss = bce(c_real, torch.ones_like(c_real)) + bce(c_fake, torch.zeros_like(c_fake))
                opt_c.zero_grad()
                c_loss.backward()
                opt_c.step()

            noise = torch.randn(real_x.size(0), config.noise_dim, device=device)
            fake_x = apply_activations(gen(noise, real_c), spec)
            c_fake_for_g = critic(fake_x, real_c)
            g_loss = -c_fake_for_g.mean() if config.loss_type == "wgan_gp" else bce(c_fake_for_g, torch.ones_like(c_fake_for_g))
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            epoch_c_loss += c_loss.item()
            epoch_g_loss += g_loss.item()
            n_batches += 1

        history["critic_loss"].append(epoch_c_loss / n_batches)
        history["gen_loss"].append(epoch_g_loss / n_batches)

    return gen, history


def save_checkpoint(path: str | Path, gen: Generator, transformer: TabularDataTransformer, cond_vocab: list[str], config: TrainConfig):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator_state": gen.state_dict(),
            "cond_vocab": cond_vocab,
            "noise_dim": config.noise_dim,
            "hidden_dim": config.hidden_dim,
            "n_blocks": config.n_blocks,
            "output_dim": transformer.output_dim,
        },
        path.with_suffix(".pt"),
    )
    with open(path.with_suffix(".transformer.pkl"), "wb") as f:
        pickle.dump(transformer, f)
