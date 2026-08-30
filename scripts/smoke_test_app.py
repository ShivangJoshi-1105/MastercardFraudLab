"""Exercises every loader/logic path the Streamlit pages use, without going through Streamlit
itself - catches missing artifacts, wrong dict keys, and import errors quickly and non-
interactively, as a first check before manually clicking through the actual UI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common

print("Loading taxonomy...")
tax = common.load_taxonomy()
assert len(tax) == 25, f"expected 25 taxonomy rows, got {len(tax)}"
print(f"  OK - {len(tax)} rows")

print("Loading defense model + metrics...")
model = common.load_defense_model()
metrics = common.load_json(common.MODELS_DIR / "defense" / "metrics.json")
print(f"  OK - recall={metrics['recall']:.3f}")

print("Loading after-loop model...")
model_after = common.load_defense_model(after_loop=True)
print("  OK")

print("Loading parquet artifacts...")
test_df = common.load_parquet("test_set.parquet")
train_df = common.load_parquet("train_set.parquet")
print(f"  OK - test={len(test_df)} rows, train={len(train_df)} rows")

print("Running a prediction + SHAP pass (mirrors Live Defense Demo page)...")
import shap
from src.defend.features import get_feature_matrix, FEATURE_COLUMNS

X_test = get_feature_matrix(test_df.sample(50, random_state=0))
probs = model.predict_proba(X_test)[:, 1]
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test.iloc[[0]])
assert shap_values.values.shape[1] == len(FEATURE_COLUMNS)
print(f"  OK - sample probs range [{probs.min():.3f}, {probs.max():.3f}]")

print("Loading tabular GAN + sampling (mirrors Generate Attacks page)...")
from src.generate.tabular_gan.sample import sample_attack
import numpy as np

demo_backbone = common.load_demo_backbone()
synth = sample_attack(
    common.MODELS_DIR / "tabular_gan" / "tabular_gan",
    demo_backbone.tabular_attack_types[0], 20,
    demo_backbone.ctx.new_account_id, demo_backbone.ctx.rng,
)
assert len(synth) == 20
print(f"  OK - sampled {len(synth)} synthetic {demo_backbone.tabular_attack_types[0]} rows")

print("Loading graph GAN + sampling one graph (mirrors Fraud Network Explorer page)...")
import torch

gen, cond_vocab, noise_dim = common.load_graph_gan()
noise = torch.randn(1, noise_dim)
cond = torch.zeros(1, len(cond_vocab))
with torch.no_grad():
    node_feats, adj, mask = gen(noise, cond)
print(f"  OK - sampled graph with {int(mask.sum().item())} active nodes")

print("Loading reports (fidelity, closed loop)...")
reports = ["tabular_fidelity.json", "graph_fidelity.json", "closed_loop.json", "gan_training_history.json"]
for r in reports:
    d = common.load_json(common.REPORTS_DIR / r)
    print(f"  OK - {r} has keys {list(d.keys())}")

print("\nALL SMOKE TESTS PASSED")
