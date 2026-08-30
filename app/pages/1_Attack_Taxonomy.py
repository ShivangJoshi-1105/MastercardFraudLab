import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_taxonomy

st.set_page_config(page_title="Attack Taxonomy", page_icon="📋", layout="wide")
st.title("📋 Pillar 1 — Attack Taxonomy")
st.caption("25 GenAI-powered payment fraud vectors, each mapped to the transactional fingerprint it actually leaves behind")

df = load_taxonomy()

col1, col2, col3 = st.columns(3)
col1.metric("Total vectors identified", len(df))
col2.metric("Concretely simulated", int(df["simulated"].sum()))
col3.metric("Categories", df["category"].nunique())

st.divider()

categories = ["All"] + sorted(df["category"].unique().tolist())
selected_category = st.selectbox("Filter by category", categories)
show_simulated_only = st.checkbox("Show simulated agents only", value=False)

filtered = df.copy()
if selected_category != "All":
    filtered = filtered[filtered["category"] == selected_category]
if show_simulated_only:
    filtered = filtered[filtered["simulated"]]

for _, row in filtered.iterrows():
    badge = "✅ Simulated" if row["simulated"] else "📄 Documented"
    with st.expander(f"**{row['id']}** — {row['name']}  ·  {badge}"):
        st.markdown(f"**Category:** {row['category']}")
        st.markdown(f"**GenAI enabler:** {row['genai_enabler']}")
        st.markdown(f"**Mechanism:** {row['mechanism']}")
        st.markdown(f"**Transactional fingerprint:** {row['fingerprint']}")
        if row["agent"]:
            st.markdown(f"**Simulated by agent:** `{row['agent']}`")

st.caption(f"Showing {len(filtered)} of {len(df)} vectors. Full narrative version: `docs/ATTACK_TAXONOMY.md`.")
