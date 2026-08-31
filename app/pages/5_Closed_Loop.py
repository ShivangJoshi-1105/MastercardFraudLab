import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import artifacts_ready, get_session_pool, load_defense_model, load_json, load_parquet, load_tabular_gan, PROCESSED_DIR, REPORTS_DIR
from src.defend.features import get_feature_matrix

st.set_page_config(page_title="Closed Loop", layout="wide")
st.title("Closing the Loop")
st.caption("The defense retrains on the attacks it just failed to catch — attacker and defender adapt to each other instead of running as three separate pillars")

if not artifacts_ready():
    st.warning("Run `python scripts/run_pipeline.py` first.")
    st.stop()

st.divider()
st.header("A. Close the loop on what you generated")
st.caption("Uses exactly the attacks accumulated in your session pool from the Generate Attacks page — nothing prebuilt.")

session_pool = get_session_pool()

if len(session_pool) == 0:
    st.info("Your session pool is empty. Go to **Generate Attacks**, generate a few incidents (any agent or GAN), then come back here.")
else:
    n_incidents = session_pool["incident_id"].nunique()
    st.markdown(f"**Session pool:** {len(session_pool)} transactions across {n_incidents} incident(s), types: {', '.join(sorted(session_pool['attack_type'].unique()))}")

    if n_incidents < 2:
        st.warning("Generate at least 2 incidents (ideally more, and more than one type) on the Generate Attacks page for a meaningful train/holdout split.")

    if st.button("Retrain on my session pool", type="primary"):
        from src.closed_loop.feedback import run_session_iteration

        with st.spinner("Retraining on your session pool (a few seconds)..."):
            model = load_defense_model()
            legit_reference = load_parquet("demo_legit_sample.parquet")
            background_train = pd.read_parquet(PROCESSED_DIR / "closed_loop_demo_train_sample.parquet")
            result = run_session_iteration(session_pool, legit_reference, background_train, model, seed=int(np.random.randint(0, 1_000_000)))
        st.session_state["session_loop_result"] = result

    if "session_loop_result" in st.session_state:
        r = st.session_state["session_loop_result"]
        st.markdown(
            f"Trained on **{r['n_train_incidents']} incident(s)** from your pool (plus a small background reference set); "
            f"held out the other **{r['n_holdout_incidents']} incident(s)** ({r['n_holdout_rows']} rows) for evaluation — "
            "the model never saw these during this retrain."
        )
        c1, c2 = st.columns(2)
        c1.metric("Detected in your held-out incidents — before", f"{r['detection_before']:.1%}")
        c2.metric(
            "Detected in your held-out incidents — after", f"{r['detection_after']:.1%}",
            delta=f"{(r['detection_after'] - r['detection_before']):+.1%}",
        )
        st.caption(
            "This is a session-scoped model trained only in memory for this demo — it never "
            "overwrites the deployed defense model, so the reference numbers in README.md and the "
            "walkthrough are unaffected by anything run here."
        )

st.divider()
st.header("B. Automatic weak-spot mining — Red-Team GAN")
st.caption("A second, independent mechanism: instead of using your generated attacks, this mines whatever the current model is already weakest against and trains a generator specifically to exploit it.")

st.markdown(
    """
1. False negatives on the held-out test set are mined to find the defense's weakest attack type.
2. A differentiable surrogate of the live XGBoost classifier is distilled in the tabular GAN's representation space.
3. A Red-Team GAN generator trains against that surrogate — its own loss rewards it for being scored as legitimate by the current defense, not only for fooling the critic.
4. The harder synthetic batch is folded into the training set and the classifier is retrained.
5. Detection rate is re-measured on a fresh, disjoint batch from the same Red-Team GAN, since aggregate test-set metrics are already near-ceiling and cannot visibly move in one round.
"""
)

if st.button("Run automatic mining iteration"):
    from src.closed_loop.feedback import run_closed_loop_iteration

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
        "Both numbers come from this run's own fresh, held-out Red-Team batch — re-running mines "
        "whatever the retrained model's new weakest spot is and produces a different batch and "
        "different numbers each time."
    )

st.divider()
st.header("Reference: full offline run")
st.caption("Same mechanism as section B, run once against the full training set — the numbers reported in README.md and the walkthrough.")

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
