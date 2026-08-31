import streamlit as st

from common import artifacts_ready

st.set_page_config(page_title="AI Defense Lab for Payment Security", layout="wide")

st.title("AI Defense Lab for Payment Security")
st.caption("Mastercard Innovation Challenge @ GFF 2026 — a closed-loop red-team/blue-team system for GenAI-powered payment fraud")

st.markdown(
    """
This prototype demonstrates a closed loop across the three pillars the challenge asks for:

| Pillar | What it does | Where |
|---|---|---|
| **1. Identify** | 25 GenAI-powered payment fraud vectors, mapped to their real-world transactional fingerprint | *Attack Taxonomy* page |
| **2. Generate** | 10 rule-based attack agents + a from-scratch tabular WGAN-GP + a hand-rolled graph GAN, generating labeled fraud at scale | *Generate Attacks* & *Fraud Network Explorer* pages |
| **3. Defend** | An XGBoost classifier on engineered velocity/graph features, with SHAP explainability | *Live Defense Demo* page |
| **Closed loop** | A "Red-Team GAN" (our own GAN modification) that mines the defense's blind spots and generates harder fraud that specifically evades it, closing the loop | *Closed Loop* page |

Use the sidebar to navigate between pages. Every page reads real artifacts produced by
`scripts/run_pipeline.py` — nothing here is mocked or hand-scripted for the demo.
"""
)

if not artifacts_ready():
    st.warning(
        "Pipeline artifacts not found yet. Run `python scripts/run_pipeline.py` from the project "
        "root first — it trains both GANs and the defense classifier and saves everything this "
        "app reads."
    )
else:
    st.success("Pipeline artifacts found. Explore the pages in the sidebar.")
