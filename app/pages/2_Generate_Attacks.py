import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import add_to_session_pool, artifacts_ready, clear_session_pool, get_session_pool, load_demo_backbone, load_json, MODELS_DIR, REPORTS_DIR

st.set_page_config(page_title="Generate Attacks", layout="wide")
st.title("Pillar 2 — Generate Attacks")
st.caption("Rule-based agents establish the ground-truth fraud pattern; two from-scratch GANs learn to scale it up")

if not artifacts_ready():
    st.warning("Run `python scripts/run_pipeline.py` first.")
    st.stop()

pool = get_session_pool()
pool_col1, pool_col2 = st.columns([3, 1])
with pool_col1:
    if len(pool):
        st.info(
            f"**Your session pool: {len(pool)} transactions** across {pool['attack_type'].nunique()} "
            f"attack type(s) ({', '.join(sorted(pool['attack_type'].unique()))}). Everything generated "
            "below is added to this pool - go to **Live Defense Demo** to score it, or **Closed Loop** "
            "to retrain on it."
        )
    else:
        st.info("Your session pool is empty. Generate some attacks below - they carry through to the Live Defense Demo and Closed Loop pages.")
with pool_col2:
    if st.button("Clear session pool", disabled=len(pool) == 0):
        clear_session_pool()
        st.rerun()

tab1, tab2, tab3 = st.tabs(["Rule-based agents (live)", "Tabular WGAN-GP", "Graph GAN"])

with tab1:
    st.subheader("Run a rule-based agent live")
    backbone = load_demo_backbone()
    all_types = backbone.tabular_attack_types + backbone.graph_attack_types
    agent_name = st.selectbox("Attack agent", all_types)
    n_incidents = st.slider("Number of incidents", 1, 50, 5)

    if st.button("Generate", key="gen_rule_based"):
        from src.generate.rule_based_agents import get_agent

        agent = get_agent(agent_name)
        df = agent.generate(backbone.ctx, n_incidents)
        add_to_session_pool(df)
        st.success(f"Generated {len(df)} transactions across {n_incidents} incidents of type `{agent_name}` — added to your session pool.")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download as CSV", df.to_csv(index=False), file_name=f"{agent_name}_sample.csv")

with tab2:
    st.subheader("Tabular WGAN-GP — behavioral fraud, learned at scale")
    fidelity = load_json(REPORTS_DIR / "tabular_fidelity.json")

    c1, c2 = st.columns(2)
    c1.metric("Discriminative AUC (0.5 = indistinguishable from real)", f"{fidelity['discriminative_auc']:.3f}")
    c2.metric("Correlation-matrix difference", f"{fidelity['correlation_diff']:.3f}")

    st.markdown("**Per-feature KS statistic** (lower = closer to the real distribution)")
    ks_df = pd.DataFrame(list(fidelity["ks_statistics"].items()), columns=["feature", "ks_statistic"])
    st.bar_chart(ks_df.set_index("feature"))

    attack_type = st.selectbox("Sample synthetic fraud of type", backbone.tabular_attack_types, key="gan_attack_type")
    n_samples = st.slider("How many synthetic rows", 10, 500, 100, key="gan_n")
    if st.button("Sample from tabular GAN"):
        from src.generate.tabular_gan.sample import sample_attack

        synth = sample_attack(MODELS_DIR / "tabular_gan" / "tabular_gan", attack_type, n_samples, backbone.ctx.new_account_id, backbone.ctx.rng)
        add_to_session_pool(synth)
        st.success(f"Sampled {len(synth)} rows — added to your session pool.")
        st.dataframe(synth, use_container_width=True)
        fig = px.histogram(synth, x="amount", nbins=40, title=f"GAN-generated amount distribution — {attack_type}")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Graph GAN — network-structured fraud (mule chains, rings, fan-in bursts)")
    st.markdown(
        "A row-independent model can't represent *who-pays-whom* topology - see the "
        "**Fraud Network Explorer** page for a visual real-vs-synthetic comparison of the graphs "
        "this model produces."
    )
    graph_fidelity = load_json(REPORTS_DIR / "graph_fidelity.json")
    cols = st.columns(4)
    cols[0].metric("Degree distribution KS", f"{graph_fidelity['degree_ks']:.3f}")
    cols[1].metric("Real avg. clustering", f"{graph_fidelity['real_avg_clustering']:.3f}")
    cols[2].metric("Synthetic avg. clustering", f"{graph_fidelity['synth_avg_clustering']:.3f}")
    if "discriminative_win_rate" in graph_fidelity:
        cols[3].metric("Critic win-rate (0.5=indistinguishable)", f"{graph_fidelity['discriminative_win_rate']:.3f}")

    if "note" in graph_fidelity:
        st.warning(
            f"**Honest engineering note:** {graph_fidelity['note']} We ran a systematic sweep "
            "of the density-regularization weight (`scripts/experiment_graph_gan.py`) and "
            "documented the failure modes rather than quietly shipping a broken generator."
        )

    st.markdown("**Generate network fraud with the rule-based topology agents instead** (correct topology, added to your session pool):")
    graph_attack_type = st.selectbox("Network attack type", backbone.graph_attack_types, key="graph_attack_type")
    n_graph_incidents = st.slider("Number of incidents", 1, 20, 3, key="graph_n")
    if st.button("Generate", key="gen_graph"):
        from src.generate.rule_based_agents import get_agent

        agent = get_agent(graph_attack_type)
        gdf = agent.generate(backbone.ctx, n_graph_incidents)
        add_to_session_pool(gdf)
        st.success(f"Generated {len(gdf)} transactions across {n_graph_incidents} incidents of type `{graph_attack_type}` — added to your session pool.")
        st.dataframe(gdf, use_container_width=True)
