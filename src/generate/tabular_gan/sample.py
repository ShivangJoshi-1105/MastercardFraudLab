"""Load a trained tabular GAN checkpoint and sample synthetic attack rows from it, conditioned
on a chosen attack type. Synthetic account ids are minted fresh (the GAN never saw or generates
identifiers - it only learns the numeric/categorical feature distribution), keeping the
generative model's job scoped to "what does this fraud type's numbers look like" rather than
also having to invent internally-consistent fake account graphs, which the rule-based agents
already do well."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data_transformer import TabularDataTransformer
from .models import Generator, apply_activations


def load_generator(path: str | Path):
    path = Path(path)
    ckpt = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=False)
    with open(path.with_suffix(".transformer.pkl"), "rb") as f:
        transformer: TabularDataTransformer = pickle.load(f)
    gen = Generator(ckpt["noise_dim"], len(ckpt["cond_vocab"]), ckpt["hidden_dim"], ckpt["output_dim"], ckpt["n_blocks"])
    gen.load_state_dict(ckpt["generator_state"])
    gen.eval()
    return gen, transformer, ckpt["cond_vocab"], ckpt["noise_dim"]


def sample_attack(path: str | Path, attack_type: str, n: int, account_minter, rng: np.random.Generator) -> pd.DataFrame:
    """`account_minter` is a callable(prefix) -> unique account id string, e.g.
    `AgentContext.new_account_id`, so GAN-sampled rows plug into the same downstream account
    bookkeeping the rule-based agents use."""
    gen, transformer, cond_vocab, noise_dim = load_generator(path)
    if attack_type not in cond_vocab:
        raise ValueError(f"GAN wasn't trained on attack type {attack_type!r}; trained on {cond_vocab}")

    cond_idx = cond_vocab.index(attack_type)
    noise = torch.randn(n, noise_dim)
    cond = torch.zeros(n, len(cond_vocab))
    cond[:, cond_idx] = 1.0

    with torch.no_grad():
        raw = gen(noise, cond)
        fake_x = apply_activations(raw, transformer.activation_spec())

    df = transformer.inverse_transform(fake_x.numpy())
    df["nameOrig"] = [account_minter("GAN") for _ in range(n)]
    df["nameDest"] = [account_minter("GAN") for _ in range(n)]
    df["isFraud"] = 1
    df["attack_type"] = attack_type
    df["incident_id"] = [f"gan_{attack_type}_{i}_{rng.integers(0, 1_000_000)}" for i in range(n)]

    from ..rule_based_agents.base import TRANSACTION_COLUMNS

    return df[TRANSACTION_COLUMNS]
