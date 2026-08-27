"""
Generator and Critic for the tabular WGAN-GP, hand-written in PyTorch.

**Why Wasserstein + gradient penalty instead of the "vanilla" GAN loss you'll see in every intro
tutorial** — this is the single most important design decision in this file, and it's worth
being explicit about it because it's also the part most from-scratch tabular GANs get wrong.
Vanilla GAN loss (binary cross-entropy, discriminator outputs a 0-1 "real" probability) is
notoriously prone to **mode collapse**: the generator finds one output the discriminator can't
easily reject and produces it for every input, ignoring the noise vector. On transaction amounts
— which are already multimodal (see `data_transformer.py`) — this failure is dramatic and easy
to demonstrate: the generator collapses onto whichever mode had the most training mass and
stops producing the rest. Train the vanilla version first (`train.py --loss vanilla` is kept
around specifically as a demonstration of this) before training the fixed version below, so the
before/after is visible rather than assumed.

The Wasserstein critic (no sigmoid — it outputs an unbounded real-valued "score", not a
probability) with a gradient penalty on interpolated real/fake samples enforces a Lipschitz
constraint that keeps training stable and gives a smoother, more informative gradient signal
back to the generator throughout training, instead of the discriminator's gradient vanishing the
moment it gets confidently good (which is exactly what starves the generator in the vanilla
setup).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return x + self.act(self.bn(self.fc(x)))


class Generator(nn.Module):
    """Maps (noise, attack-type condition) -> a raw pre-activation vector; the caller applies
    the transformer's per-block softmax/tanh activations (see `apply_activations` below), since
    the generator itself shouldn't need to know column semantics, only column widths."""

    def __init__(self, noise_dim: int, cond_dim: int, hidden_dim: int, output_dim: int, n_blocks: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(noise_dim + cond_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, cond], dim=1)
        x = torch.relu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class Critic(nn.Module):
    """The 'discriminator' in WGAN terminology - outputs an unbounded real-valued score (higher
    = 'looks more real'), not a 0-1 probability, which is what makes the Wasserstein loss below
    valid."""

    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(input_dim + cond_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(0.3)]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, cond], dim=1))


def apply_activations(raw: torch.Tensor, activation_spec: list[tuple[int, int, str]]) -> torch.Tensor:
    """Applies softmax to each one-hot mode/category block and tanh to each continuous scalar,
    exactly the blocks `TabularDataTransformer.activation_spec()` describes. This has to happen
    per-block, not once globally - a single softmax over the whole row would incorrectly force
    every column to compete with every other column for probability mass."""
    pieces = []
    for start, width, kind in activation_spec:
        block = raw[:, start : start + width]
        if kind == "softmax":
            pieces.append(torch.softmax(block, dim=1))
        else:
            pieces.append(torch.tanh(block))
    return torch.cat(pieces, dim=1)


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor, cond: torch.Tensor, device) -> torch.Tensor:
    """Standard WGAN-GP penalty: pushes the critic's gradient norm on interpolated real/fake
    points toward 1, enforcing the 1-Lipschitz constraint the Wasserstein distance requires. This
    is what actually stabilizes training relative to vanilla weight clipping (the original WGAN's
    fix, which itself under/over-clips too easily to trust in a hand-rolled implementation)."""
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, device=device)
    interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(interpolated, cond)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()
