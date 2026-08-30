"""
The closed-loop iteration: mine the current defense's blind spots, generate a harder adversarial
batch that specifically targets them via the Red-Team GAN, retrain the defense on the enlarged
dataset, and report whether it actually got better. This is the concrete artifact for "the
gaps your defense reveals feed back into new attack ideas" - not a metaphor, a runnable function
that measurably closes the loop.

Steps:
1. Mine false negatives from the current model's test-set predictions (`evaluate.
   false_negatives_mask`) to see which attack type(s) are slipping through most.
2. Distill a differentiable surrogate of the *current* classifier (see `defense_aware_gan.py`)
   in the tabular GAN's representation space.
3. Train the Red-Team GAN generator against that surrogate, conditioned on the worst-performing
   attack type, producing a batch explicitly optimized to be missed by the current defense.
4. Fold that harder batch into the training set and retrain XGBoost from scratch.
5. Re-evaluate on the *same* held-out test set as before, so the before/after comparison is
   apples-to-apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..defend.evaluate import evaluate_predictions, false_negatives_mask
from ..defend.features import engineer_features, get_feature_matrix
from ..defend.train_classifier import train_xgboost
from ..generate.defense_aware_gan import distill_surrogate, train_redteam_generator
from ..generate.tabular_gan.data_transformer import TabularDataTransformer
from ..generate.tabular_gan.train import TrainConfig


def worst_attack_type(test_df: pd.DataFrame, y_prob: np.ndarray, threshold: float = 0.5) -> str:
    fn_mask = false_negatives_mask(test_df["label"].to_numpy(), y_prob, threshold)
    fraud_only = test_df[test_df["label"] == 1].reset_index(drop=True)
    fn_only = fn_mask[test_df["label"].to_numpy() == 1]
    if fn_only.sum() == 0:
        return fraud_only["attack_type"].mode().iloc[0]  # nothing missed - just pick the majority type
    return fraud_only.loc[fn_only, "attack_type"].mode().iloc[0]


def run_closed_loop_iteration(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
    y_prob_before: np.ndarray,
    transformer: TabularDataTransformer,
    tabular_cond_vocab: list[str],
    account_minter,
    rng: np.random.Generator,
    n_harder_samples: int = 500,
    gan_epochs: int = 60,
):
    metrics_before = evaluate_predictions(test_df["label"].to_numpy(), y_prob_before)
    target_type = worst_attack_type(test_df, y_prob_before)
    if target_type not in tabular_cond_vocab:
        target_type = tabular_cond_vocab[0]  # graph-only false negatives fall back to any tabular type

    # 2. Distill a surrogate of the *current* model in the GAN's representation space.
    fit_cols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    X_transformed = torch.tensor(transformer.transform(train_df[fit_cols]))
    soft_labels = torch.tensor(model.predict_proba(get_feature_matrix(train_df))[:, 1].astype(np.float32))
    surrogate = distill_surrogate(X_transformed, soft_labels)

    # 3. Train the Red-Team generator against it, conditioned on the type the model misses most.
    attack_only = train_df[train_df["attack_type"] == target_type]
    config = TrainConfig(epochs=gan_epochs, batch_size=min(128, len(attack_only)) or 8)
    redteam_gen, gan_history = train_redteam_generator(attack_only, tabular_cond_vocab, transformer, surrogate, config)

    # 4. Sample a harder batch and fold it into a retrained classifier.
    from ..generate.tabular_gan.models import apply_activations

    noise = torch.randn(n_harder_samples, config.noise_dim)
    cond = torch.zeros(n_harder_samples, len(tabular_cond_vocab))
    cond[:, tabular_cond_vocab.index(target_type)] = 1.0
    with torch.no_grad():
        fake_x = apply_activations(redteam_gen(noise, cond), transformer.activation_spec())
    harder_df = transformer.inverse_transform(fake_x.numpy())
    harder_df["nameOrig"] = [account_minter("RT") for _ in range(n_harder_samples)]
    harder_df["nameDest"] = [account_minter("RT") for _ in range(n_harder_samples)]
    harder_df["isFraud"] = 1
    harder_df["attack_type"] = target_type
    harder_df["incident_id"] = [f"redteam_{target_type}_{i}" for i in range(n_harder_samples)]
    harder_engineered = engineer_features(harder_df)
    harder_engineered["label"] = 1

    retrain_df = pd.concat([train_df, harder_engineered], ignore_index=True)
    new_model = train_xgboost(retrain_df)

    y_prob_after = new_model.predict_proba(get_feature_matrix(test_df))[:, 1]
    metrics_after = evaluate_predictions(test_df["label"].to_numpy(), y_prob_after)

    return {
        "target_attack_type": target_type,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "gan_history": gan_history,
        "new_model": new_model,
        "harder_samples": harder_df,
    }
