import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import artifacts_ready, load_defense_model, load_json, load_parquet, load_tabular_gan, PROCESSED_DIR, REPORTS_DIR
from src.defend.features import get_feature_matrix

st.set_page_config(page_title="Closed Loop", layout="wide")
st.title("Closing the Loop — Red-Team GAN")
st.caption("The generator is rewarded for evading the current defense, not just for fooling the critic — this is what makes the system a loop instead of three separate pillars")

if not artifacts_ready():
    st.warning("Run `python scripts/run_pipeline.py` first.")
    st.stop()

st.markdown(
    """
**Mechanism:**
1. The defense's false negatives on the held-out test set are mined to find its weakest attack type.
2. A differentiable surrogate of the live XGBoost classifier is distilled in the tabular GAN's representation space.
3. The Red-Team GAN generator trains against that surrogate, so its output is explicitly optimized to be scored as legitimate by the current defense.
4. The harder synthetic batch is folded into the training set and the classifier is retrained.
5. Detection rate is re-measured on a fresh, disjoint batch from the same Red-Team GAN — fraud built to evade the pre-iteration model — so the before/after comparison tests whether retraining actually closed the gap, not just aggregate metrics that are already near-ceiling.
"""
)

st.divider()
st.subheader("Run an iteration now")
st.caption("Runs live: mines the current model's weak spot, trains a Red-Team GAN generator against it, retrains the classifier, and measures the result — same mechanism as the full pipeline, on a smaller sample so it completes in under a minute.")

if st.button("Run closed-loop iteration", type="primary"):
    from src.closed_loop.feedback import run_closed_loop_iteration
    from src.generate.tabular_gan.train import TrainConfig

    with st.spinner("Mining weak spots, training Red-Team GAN, retraining classifier..."):
        model = load_defense_model()
        test_df = load_parquet("test_set.parquet")
        train_sample = pd.read_parquet(PROCESSED_DIR / "closed_loop_demo_train_sample.parquet")
        _, transformer, cond_vocab, _ = load_tabular_gan()

        rng = np.random.default_rng()
        counter = {"n": 0}

        def account_minter(prefix):
            counter["n"] += 1
            return f"{prefix}LIVE{counter['n']}{int(rng.integers(0, 999999))}"

        y_prob_test_before = model.predict_proba(get_feature_matrix(test_df))[:, 1]
        result = run_closed_loop_iteration(
            train_sample, test_df, model, y_prob_test_before, transformer, cond_vocab,
            account_minter, rng, n_harder_samples=150, gan_epochs=25,
        )
    st.session_state["live_closed_loop_result"] = result
    st.success(f"Done. Targeted weakest attack type: `{result['target_attack_type']}`")

if "live_closed_loop_result" in st.session_state:
    r = st.session_state["live_closed_loop_result"]
    st.markdown(f"**This run targeted:** `{r['target_attack_type']}`")
    c1, c2 = st.columns(2)
    c1.metric("Detected before this iteration", f"{r['holdout_detection_rate_before']:.1%}")
    c2.metric(
        "Detected after this iteration", f"{r['holdout_detection_rate_after']:.1%}",
        delta=f"{(r['holdout_detection_rate_after'] - r['holdout_detection_rate_before']):+.1%}",
    )
    st.caption(
        "Both numbers come from this run's own fresh, held-out Red-Team batch — re-running the "
        "button above mines whatever the retrained model's new weakest spot is and produces a "
        "different batch and different numbers, since the loop keeps adapting each time."
    )

st.divider()
st.subheader("Reference: full offline run (documented in README.md and the walkthrough)")
st.caption("Same mechanism, run once against the full training set for the numbers reported in the write-up.")

result = load_json(REPORTS_DIR / "closed_loop.json")

if "holdout_detection_rate_before" in result:
    c1, c2 = st.columns(2)
    c1.metric("Detected before retraining", f"{result['holdout_detection_rate_before']:.1%}")
    c2.metric(
        "Detected after retraining", f"{result['holdout_detection_rate_after']:.1%}",
        delta=f"{(result['holdout_detection_rate_after'] - result['holdout_detection_rate_before']):+.1%}",
    )

before, after = result["metrics_before"], result["metrics_after"]
metric_names = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
comparison = pd.DataFrame(
    {
        "metric": metric_names * 2,
        "value": [before[m] for m in metric_names] + [after[m] for m in metric_names],
        "when": ["before"] * len(metric_names) + ["after"] * len(metric_names),
    }
)
fig = px.bar(comparison, x="metric", y="value", color="when", barmode="group", title="Aggregate held-out test-set metrics, before vs. after")
st.plotly_chart(fig, use_container_width=True)

cols = st.columns(3)
cols[0].metric("Recall - before", f"{before['recall']:.3f}")
cols[1].metric("Recall - after", f"{after['recall']:.3f}", delta=f"{after['recall'] - before['recall']:+.3f}")
cols[2].metric("FPR on legit - after", f"{after['false_positive_rate_on_legit']:.4f}", delta=f"{after['false_positive_rate_on_legit'] - before['false_positive_rate_on_legit']:+.4f}", delta_color="inverse")

if "redteam_gan_history" in result:
    st.divider()
    st.subheader("Red-Team GAN training curve (full offline run)")
    rt = result["redteam_gan_history"]
    rt_df = pd.DataFrame({"epoch": range(len(rt["evasion_loss"])), "evasion_loss": rt["evasion_loss"], "critic_loss": rt["critic_loss"]})
    st.line_chart(rt_df.set_index("epoch"))
    st.caption("Falling evasion loss means the generator is getting better at producing fraud the surrogate defense scores as legitimate.")
