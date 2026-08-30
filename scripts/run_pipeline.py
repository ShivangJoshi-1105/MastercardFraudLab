"""
Runs the full non-UI pipeline end-to-end: Identify (loads the taxonomy) -> Generate (rule-based
agents, tabular GAN, graph GAN) -> Defend (feature engineering, XGBoost, SHAP) -> Closed loop
(Red-Team GAN iteration). Every artifact the Streamlit app needs is written to `models/` and
`data/processed/` by the end of this script, so the app never has to retrain anything live - it
just loads what this script produced.

Run with: .venv\\Scripts\\python.exe scripts\\run_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generate.simulate import Backbone, generate_all_attacks, load_backbone, save_labeled_dataset
from src.generate.rule_based_agents import get_agent
from src.generate.tabular_gan.data_transformer import TabularDataTransformer
from src.generate.tabular_gan.train import TrainConfig, save_checkpoint, train_tabular_gan
from src.generate.tabular_gan.sample import sample_attack
from src.generate.fidelity_eval import fidelity_report
from src.generate.graph_gan.graph_data import incidents_to_tensors, MAX_NODES
from src.generate.graph_gan.train import GraphTrainConfig, save_graph_checkpoint, train_graph_gan
from src.generate.graph_gan.graph_fidelity_eval import graph_fidelity_report
from src.defend.train_classifier import train_and_evaluate, save_model
from src.defend.features import engineer_features, get_feature_matrix
from src.closed_loop.feedback import run_closed_loop_iteration

MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
FIT_COLS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("Step 1/6 - Loading PaySim backbone and computing real-data statistics...")
    backbone = load_backbone()
    log(f"  legit sample: {len(backbone.legit_df):,} rows | native PaySim fraud: {len(backbone.native_fraud_df):,} rows")
    log(f"  tabular attack types: {backbone.tabular_attack_types}")
    log(f"  graph attack types: {backbone.graph_attack_types}")

    log("Step 2/6 - Running all 10 rule-based attack agents at scale...")
    attacks_df = generate_all_attacks(backbone, incidents_per_agent=150)
    save_labeled_dataset(backbone, attacks_df, PROCESSED_DIR)
    log(f"  generated {len(attacks_df):,} labeled fraud rows across {attacks_df['incident_id'].nunique():,} incidents")

    # ---------------------------------------------------------------- Tabular GAN
    log("Step 3/6 - Training the tabular WGAN-GP on behavioral attack agents' output...")
    tabular_df = attacks_df[attacks_df["attack_type"].isin(backbone.tabular_attack_types)].reset_index(drop=True)
    transformer = TabularDataTransformer().fit(tabular_df[FIT_COLS])
    tab_config = TrainConfig(epochs=100, batch_size=256, loss_type="wgan_gp")
    tab_gen, tab_history = train_tabular_gan(tabular_df, backbone.tabular_attack_types, transformer, tab_config)
    save_checkpoint(MODELS_DIR / "tabular_gan" / "tabular_gan", tab_gen, transformer, backbone.tabular_attack_types, tab_config)
    log(f"  final critic loss: {tab_history['critic_loss'][-1]:.3f} | final gen loss: {tab_history['gen_loss'][-1]:.3f}")

    synth_tabular_frames = []
    for attack_type in backbone.tabular_attack_types:
        synth_tabular_frames.append(
            sample_attack(MODELS_DIR / "tabular_gan" / "tabular_gan", attack_type, n=300, account_minter=backbone.ctx.new_account_id, rng=backbone.ctx.rng)
        )
    synth_tabular_df = pd.concat(synth_tabular_frames, ignore_index=True)

    tab_fidelity = fidelity_report(tabular_df, synth_tabular_df, ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
    log(f"  tabular fidelity - discriminative AUC: {tab_fidelity['discriminative_auc']:.3f} (0.5 = indistinguishable)")

    # ---------------------------------------------------------------- Graph GAN
    log("Step 4/6 - Training the hand-rolled graph GAN on network attack agents' output...")
    graph_df = attacks_df[attacks_df["attack_type"].isin(backbone.graph_attack_types)].reset_index(drop=True)
    node_feats, adj, mask, labels = incidents_to_tensors(graph_df, backbone.graph_attack_types)
    # Epoch count kept modest deliberately: the sweep in scripts/experiment_graph_gan.py showed
    # this hand-rolled generator doesn't converge to realistic topology regardless of epoch count
    # in this timeframe (see the fallback note below), and an earlier run at 200 epochs stalled
    # for over an hour on this machine, most likely due to external CPU contention rather than
    # the model itself - capping epochs bounds that risk without changing the documented outcome.
    graph_config = GraphTrainConfig(epochs=60, batch_size=32, sparsity_weight=0.02)
    graph_gen, graph_critic, graph_history = train_graph_gan(node_feats, adj, mask, labels, backbone.graph_attack_types, graph_config)
    save_graph_checkpoint(MODELS_DIR / "graph_gan" / "graph_gan", graph_gen, backbone.graph_attack_types, graph_config)
    log(f"  final critic loss: {graph_history['critic_loss'][-1]:.3f} | final gen loss: {graph_history['gen_loss'][-1]:.3f}")

    with torch.no_grad():
        noise = torch.randn(len(node_feats), graph_config.noise_dim)
        cond_idx_map = {a: i for i, a in enumerate(backbone.graph_attack_types)}
        cond = torch.zeros(len(labels), len(backbone.graph_attack_types))
        for i, lab in enumerate(labels):
            cond[i, cond_idx_map[lab]] = 1.0
        synth_feats, synth_adj, synth_mask = graph_gen(noise, cond)
    graph_fidelity = graph_fidelity_report(node_feats, adj, mask, synth_feats, synth_adj, synth_mask, critic=graph_critic, real_cond=cond, synth_cond=cond)
    log(f"  graph GAN topology fidelity - degree KS: {graph_fidelity['degree_ks']:.3f}, cycle rate real/synth: {graph_fidelity['real_cycle_rate']:.2f}/{graph_fidelity['synth_cycle_rate']:.2f}")

    # Documented fallback (see docs/ATTACK_TAXONOMY.md / build plan risk note): across a
    # systematic sweep of the density-regularization weight, the hand-rolled graph generator
    # converged to one of two degenerate solutions (fully-connected or empty graphs) rather than
    # learning real mule/ring/fan-in topology within this project's time budget. Rather than feed
    # a defense classifier known-wrong topology, the *training* data uses a larger batch from the
    # already-correct rule-based graph agents; the trained GraphGenerator/GraphDiscriminator are
    # still real, working artifacts (used for the app's live demo and as the fidelity scorer
    # above) - this is an explicit scope decision, not a hidden gap.
    graph_fidelity["note"] = (
        "Graph GENERATOR did not converge to realistic topology within the project timeline "
        "(see sweep in scripts/experiment_graph_gan.py); synthetic graph-attack training data "
        "below is sourced from the rule-based agents at scale instead. The discriminator/critic "
        "trained normally and is reported above as a genuine fidelity scorer."
    )
    log("  using rule-based graph agents (scaled up) for training-data augmentation instead of GAN samples - see note in graph_fidelity.json")
    graph_scaleup_frames = []
    for name in backbone.graph_attack_types:
        agent = get_agent(name)
        graph_scaleup_frames.append(agent.generate(backbone.ctx, 200))
    synth_graph_df = pd.concat(graph_scaleup_frames, ignore_index=True)

    # ---------------------------------------------------------------- Defense classifier
    log("Step 5/6 - Engineering features and training the XGBoost defense classifier...")
    legit_labeled = backbone.legit_df.assign(attack_type="legit", incident_id="legit")
    combined_df = pd.concat([legit_labeled, attacks_df, synth_tabular_df, synth_graph_df], ignore_index=True)
    model, train_df, test_df, y_prob, metrics = train_and_evaluate(combined_df)
    save_model(model, metrics, MODELS_DIR / "defense")
    log(f"  precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
        f"roc_auc={metrics['roc_auc']:.3f} FPR-on-legit={metrics['false_positive_rate_on_legit']:.4f}")

    # ---------------------------------------------------------------- Closed loop
    log("Step 6/6 - Running one closed-loop iteration (Red-Team GAN -> retrain -> compare)...")
    loop_result = run_closed_loop_iteration(
        train_df, test_df, model, y_prob, transformer, backbone.tabular_attack_types,
        account_minter=backbone.ctx.new_account_id, rng=backbone.ctx.rng, n_harder_samples=500, gan_epochs=60,
    )
    log(f"  targeted weakest attack type: {loop_result['target_attack_type']}")
    log(f"  aggregate test-set recall before: {loop_result['metrics_before']['recall']:.3f} -> after: {loop_result['metrics_after']['recall']:.3f}")
    log(f"  detection rate on held-out Red-Team batch before: {loop_result['holdout_detection_rate_before']:.3f} -> after: {loop_result['holdout_detection_rate_after']:.3f}")
    loop_result["new_model"].save_model(str(MODELS_DIR / "defense" / "fraud_classifier_after_loop.json"))
    loop_result["harder_samples"].to_parquet(PROCESSED_DIR / "redteam_harder_samples.parquet")

    # ---------------------------------------------------------------- Persist everything the app needs
    (MODELS_DIR / "reports").mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "reports" / "tabular_fidelity.json", "w") as f:
        json.dump(tab_fidelity, f, indent=2)
    with open(MODELS_DIR / "reports" / "graph_fidelity.json", "w") as f:
        json.dump({k: v for k, v in graph_fidelity.items()}, f, indent=2)
    with open(MODELS_DIR / "reports" / "gan_training_history.json", "w") as f:
        json.dump({"tabular": tab_history, "graph": graph_history}, f, indent=2)
    with open(MODELS_DIR / "reports" / "closed_loop.json", "w") as f:
        json.dump(
            {
                "target_attack_type": loop_result["target_attack_type"],
                "metrics_before": loop_result["metrics_before"],
                "metrics_after": loop_result["metrics_after"],
                "holdout_detection_rate_before": loop_result["holdout_detection_rate_before"],
                "holdout_detection_rate_after": loop_result["holdout_detection_rate_after"],
                "redteam_gan_history": loop_result["gan_history"],
            },
            f, indent=2,
        )
    test_df.to_parquet(PROCESSED_DIR / "test_set.parquet")
    train_df.to_parquet(PROCESSED_DIR / "train_set.parquet")
    synth_tabular_df.to_parquet(PROCESSED_DIR / "synthetic_tabular_attacks.parquet")
    if len(synth_graph_df):
        synth_graph_df.to_parquet(PROCESSED_DIR / "synthetic_graph_attacks.parquet")

    log("Pipeline complete. Artifacts saved under models/ and data/processed/.")
    log("Next: streamlit run app/Home.py")


if __name__ == "__main__":
    main()
