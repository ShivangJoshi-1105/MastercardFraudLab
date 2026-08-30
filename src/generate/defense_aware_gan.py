"""
The "Red-Team GAN" — our own modification to the GAN objective, and the piece that actually
makes this project a *closed loop* instead of three independent pillars bolted together.

**The idea.** A standard GAN generator has one job: produce samples the critic can't distinguish
from real ones. That alone gives you realistic fraud, but realistic isn't the same as *dangerous*
— it says nothing about whether our own defense model would actually miss it. The Red-Team GAN
adds a second objective: the generator is also rewarded for producing fraud that a snapshot of
our *live* XGBoost defense classifier scores as legitimate. Every closed-loop iteration
re-snapshots the classifier after it's retrained, so the next round of generated fraud has to
evade the improved model, not the stale one — a genuine adversarial arms race between the
generator and the defense, mirroring exactly the "attacks train the defense; the defense's gaps
drive the next round of attacks" framing the competition brief uses.

**Why a surrogate, not the real XGBoost model, in the loss.** XGBoost is a tree ensemble — it has
no gradient with respect to its input that PyTorch's autograd can use, so it cannot sit directly
inside the generator's backward pass. The standard practical workaround for attacking a
non-differentiable model (used throughout adversarial ML research) is **surrogate distillation**:
train a small differentiable model to mimic the real classifier's *probability outputs*, then
attack the surrogate — an attack that fools a good surrogate transfers to the real model far more
often than a random perturbation would. `SurrogateDefense` is trained on the GAN's own
transformed feature space (the same tensor the generator already outputs), specifically so the
evasion loss below is a plain, fully differentiable term with no lossy round-trip through pandas
in between.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from .tabular_gan.data_transformer import TabularDataTransformer
from .tabular_gan.models import Critic, Generator, apply_activations, gradient_penalty
from .tabular_gan.train import TrainConfig


class SurrogateDefense(nn.Module):
    """A small MLP distilled to mimic the live XGBoost classifier's fraud probability, operating
    on the tabular GAN's transformed representation rather than the full engineered feature set
    (attacking the raw representation and evaluating transfer against the real pipeline is the
    standard compromise here - see module docstring)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logit; caller applies sigmoid/BCEWithLogits as needed


def distill_surrogate(transformed_X, soft_labels, epochs: int = 30, lr: float = 1e-3) -> SurrogateDefense:
    """`transformed_X`: (N, D) tensor in the tabular GAN's representation. `soft_labels`: (N,)
    tensor of the live XGBoost model's predicted fraud probability for those same rows - this is
    the distillation target, i.e. the surrogate learns to imitate the *real* defense's judgment,
    not ground-truth labels directly."""
    surrogate = SurrogateDefense(transformed_X.shape[1])
    opt = Adam(surrogate.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(transformed_X, soft_labels), batch_size=256, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader:
            pred = surrogate(xb).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)  # frozen snapshot: only the generator learns from here on
    return surrogate


def train_redteam_generator(
    real_df,
    cond_vocab: list[str],
    transformer: TabularDataTransformer,
    surrogate: SurrogateDefense,
    config: TrainConfig,
    evasion_weight: float = 1.0,
):
    """Same WGAN-GP loop as `tabular_gan/train.py`, with one addition: the generator's loss gets
    an extra `evasion_weight * evasion_loss` term pushing its output toward whatever the frozen
    surrogate currently classifies as legitimate. Everything else (critic training, gradient
    penalty) is identical, since only the generator's objective needs to change for this to
    become "red-team" training - the critic's job (tell real from fake) is unchanged."""
    device = torch.device(config.device)
    X = transformer.transform(real_df)
    cond_idx = {a: i for i, a in enumerate(cond_vocab)}
    labels = real_df["attack_type"].map(cond_idx).to_numpy()
    import numpy as np

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

    history = {"critic_loss": [], "gen_loss": [], "evasion_loss": []}

    for epoch in range(config.epochs):
        epoch_c, epoch_g, epoch_e, n_batches = 0.0, 0.0, 0.0, 0
        for real_x, real_c in loader:
            real_x, real_c = real_x.to(device), real_c.to(device)

            for _ in range(config.critic_iters):
                noise = torch.randn(real_x.size(0), config.noise_dim, device=device)
                fake_x = apply_activations(gen(noise, real_c), spec)
                c_real = critic(real_x, real_c)
                c_fake = critic(fake_x.detach(), real_c)
                gp = gradient_penalty(critic, real_x, fake_x.detach(), real_c, device)
                c_loss = c_fake.mean() - c_real.mean() + 10.0 * gp
                opt_c.zero_grad()
                c_loss.backward()
                opt_c.step()

            noise = torch.randn(real_x.size(0), config.noise_dim, device=device)
            fake_x = apply_activations(gen(noise, real_c), spec)
            adv_loss = -critic(fake_x, real_c).mean()

            evasion_logit = surrogate(fake_x).squeeze(-1)
            evasion_loss = nn.functional.binary_cross_entropy_with_logits(evasion_logit, torch.zeros_like(evasion_logit))

            g_loss = adv_loss + evasion_weight * evasion_loss
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            epoch_c += c_loss.item()
            epoch_g += adv_loss.item()
            epoch_e += evasion_loss.item()
            n_batches += 1

        history["critic_loss"].append(epoch_c / n_batches)
        history["gen_loss"].append(epoch_g / n_batches)
        history["evasion_loss"].append(epoch_e / n_batches)

    return gen, history
