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
4. Fold one harder batch into the training set and retrain XGBoost from scratch.
5. Re-evaluate on the *same* held-out test set as before (apples-to-apples), AND on a *second*,
   disjoint held-out batch sampled from the same Red-Team GAN. That second number is the honest
   headline metric: aggregate test-set recall is usually already high enough (>0.99 in our runs)
   that a single retraining round can't visibly move it - a ceiling effect, not a failure of the
   loop. The held-out red-team batch has no such ceiling: it's specifically new fraud designed to
   evade the *before* model, so "before" should catch relatively little of it, and "after" should
   catch substantially more, because the improvement it needed to make is exactly what this batch
   tests for.
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


def run_session_iteration(
    session_pool: pd.DataFrame,
    legit_reference: pd.DataFrame,
    background_train: pd.DataFrame,
    model,
    threshold: float = 0.5,
    seed: int = 0,
):
    """
    A second, faster closed-loop path driven directly by whatever a user generated in the app's
    Generate Attacks page, rather than by internally-mined weak spots. This is what makes the
    Closed Loop page's headline demo literal: attacks a person just produced become the training
    signal, and the same model architecture is retrained and re-measured against a held-out half
    of that exact pool - not against a canned batch.

    `session_pool` / `legit_reference` are raw PaySim-schema rows (not yet feature-engineered).
    `background_train` is a small, pre-engineered general sample folded in alongside the user's
    attacks so the retrain isn't learning from a handful of rows in isolation, which would
    otherwise make the "after" model a narrow memorizer rather than a generalizing classifier.

    Split is by `incident_id` (not row) so a multi-row incident - a mule chain, a bust-out
    lifecycle - never has some of its own rows in the training half and others in the held-out
    half, which would leak.
    """
    rng = np.random.default_rng(seed)
    incident_ids = np.array(session_pool["incident_id"].unique(), dtype=object)
    rng.shuffle(incident_ids)
    split = max(1, len(incident_ids) // 2)
    train_incidents, holdout_incidents = set(incident_ids[:split]), set(incident_ids[split:])
    if not holdout_incidents:  # only one incident total - test on it, still train on something
        holdout_incidents, train_incidents = train_incidents, set()

    train_half = session_pool[session_pool["incident_id"].isin(train_incidents)]
    holdout_half = session_pool[session_pool["incident_id"].isin(holdout_incidents)]

    holdout_engineered = engineer_features(holdout_half)
    y_prob_before = model.predict_proba(get_feature_matrix(holdout_engineered))[:, 1]
    detection_before = float((y_prob_before >= threshold).mean())

    legit_engineered = engineer_features(legit_reference).assign(attack_type="legit", incident_id="legit", label=0)
    train_engineered = engineer_features(train_half).assign(label=1) if len(train_half) else train_half
    retrain_df = pd.concat([background_train, legit_engineered, train_engineered], ignore_index=True)
    new_model = train_xgboost(retrain_df)

    y_prob_after = new_model.predict_proba(get_feature_matrix(holdout_engineered))[:, 1]
    detection_after = float((y_prob_after >= threshold).mean())

    return {
        "n_train_incidents": len(train_incidents),
        "n_holdout_incidents": len(holdout_incidents),
        "n_holdout_rows": len(holdout_half),
        "detection_before": detection_before,
        "detection_after": detection_after,
        "new_model": new_model,
    }


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

    # 4. Sample two disjoint batches from the trained Red-Team generator: one folded into
    # retraining, one kept aside purely for the before/after "did this actually help" check.
    from ..generate.tabular_gan.models import apply_activations

    def sample_redteam_batch(n: int, id_prefix: str) -> pd.DataFrame:
        noise = torch.randn(n, config.noise_dim)
        cond = torch.zeros(n, len(tabular_cond_vocab))
        cond[:, tabular_cond_vocab.index(target_type)] = 1.0
        with torch.no_grad():
            fake_x = apply_activations(redteam_gen(noise, cond), transformer.activation_spec())
        df = transformer.inverse_transform(fake_x.numpy())
        df["nameOrig"] = [account_minter("RT") for _ in range(n)]
        df["nameDest"] = [account_minter("RT") for _ in range(n)]
        df["isFraud"] = 1
        df["attack_type"] = target_type
        df["incident_id"] = [f"{id_prefix}_{target_type}_{i}" for i in range(n)]
        return df

    harder_df = sample_redteam_batch(n_harder_samples, "redteam_train")
    holdout_df = sample_redteam_batch(n_harder_samples, "redteam_holdout")

    harder_engineered = engineer_features(harder_df)
    harder_engineered["label"] = 1
    holdout_engineered = engineer_features(holdout_df)

    retrain_df = pd.concat([train_df, harder_engineered], ignore_index=True)
    new_model = train_xgboost(retrain_df)

    y_prob_after = new_model.predict_proba(get_feature_matrix(test_df))[:, 1]
    metrics_after = evaluate_predictions(test_df["label"].to_numpy(), y_prob_after)

    # Detection rate on the held-out adversarial batch (every row is fraud, so this is just recall).
    holdout_X = get_feature_matrix(holdout_engineered)
    detection_before = float((model.predict_proba(holdout_X)[:, 1] >= 0.5).mean())
    detection_after = float((new_model.predict_proba(holdout_X)[:, 1] >= 0.5).mean())

    return {
        "target_attack_type": target_type,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "holdout_detection_rate_before": detection_before,
        "holdout_detection_rate_after": detection_after,
        "gan_history": gan_history,
        "new_model": new_model,
        "harder_samples": harder_df,
    }
