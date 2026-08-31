import sys
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import artifacts_ready, load_demo_backbone, load_graph_gan

st.set_page_config(page_title="Fraud Network Explorer", layout="wide")
st.title("Fraud Network Explorer")
st.caption("Mule chains, collusive rings, and fan-in bursts are topology, not row-level anomalies — this page shows the shape directly")

if not artifacts_ready():
    st.warning("Run `python scripts/run_pipeline.py` first.")
    st.stop()


def draw_graph(adj, mask, title: str):
    n = int(mask.sum())
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] > 0.3:
                g.add_edge(i, j, weight=float(adj[i, j]))

    if n == 0 or g.number_of_edges() == 0:
        st.info(f"{title}: degenerate sample (no edges above threshold), try again.")
        return

    pos = nx.spring_layout(g, seed=42)
    edge_x, edge_y = [], []
    for u, v in g.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"), mode="lines", hoverinfo="none")

    node_x = [pos[i][0] for i in g.nodes()]
    node_y = [pos[i][1] for i in g.nodes()]
    degrees = [g.degree(i) for i in g.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=[str(i) for i in g.nodes()], textposition="top center",
        marker=dict(size=[12 + 4 * d for d in degrees], color=degrees, colorscale="Reds", showscale=False),
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=title, showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=400)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(fig, use_container_width=True)


backbone = load_demo_backbone()
attack_type = st.selectbox("Ring type", backbone.graph_attack_types)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Real (rule-based agent)")
    if st.button("Generate a real example", key="real_graph"):
        from src.generate.rule_based_agents import get_agent
        from src.generate.graph_gan.graph_data import incident_to_graph

        agent = get_agent(attack_type)
        df = agent.generate(backbone.ctx, 1)
        node_feats, adj, mask = incident_to_graph(df)
        draw_graph(adj, mask, f"Real: {attack_type}")
        st.dataframe(df, use_container_width=True)

with col2:
    st.subheader("Synthetic (graph GAN)")
    if st.button("Sample from graph GAN", key="synth_graph"):
        gen, cond_vocab, noise_dim = load_graph_gan()
        cond_idx = cond_vocab.index(attack_type)
        noise = torch.randn(1, noise_dim)
        cond = torch.zeros(1, len(cond_vocab))
        cond[0, cond_idx] = 1.0
        with torch.no_grad():
            node_feats, adj, mask = gen(noise, cond)
        draw_graph(adj[0].numpy(), mask[0].numpy(), f"Synthetic: {attack_type}")

st.divider()
st.markdown(
    "**Why this matters:** no single transaction in a mule chain or collusive ring looks unusual "
    "on its own — it's the shape (a chain, a cycle, a star) that gives it away, which is exactly "
    "what a row-independent tabular model cannot represent and the graph GAN was built for."
)
