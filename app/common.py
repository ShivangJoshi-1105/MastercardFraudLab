"""Shared loaders for the Streamlit app - every page reads artifacts `scripts/run_pipeline.py`
already produced rather than retraining anything live, so the app stays fast and demo-safe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = MODELS_DIR / "reports"


@st.cache_data
def load_taxonomy() -> pd.DataFrame:
    from src.identify.taxonomy import load_taxonomy as _load

    return _load()


@st.cache_data
def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_defense_model(after_loop: bool = False) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    filename = "fraud_classifier_after_loop.json" if after_loop else "fraud_classifier.json"
    model.load_model(str(MODELS_DIR / "defense" / filename))
    return model


@st.cache_data
def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / name)


@st.cache_resource
def load_tabular_gan():
    from src.generate.tabular_gan.sample import load_generator

    return load_generator(MODELS_DIR / "tabular_gan" / "tabular_gan")


@st.cache_resource
def load_graph_gan():
    from src.generate.graph_gan.sample import load_graph_generator

    return load_graph_generator(MODELS_DIR / "graph_gan" / "graph_gan")


def artifacts_ready() -> bool:
    return (MODELS_DIR / "defense" / "fraud_classifier.json").exists()


SESSION_POOL_KEY = "session_attack_pool"


def add_to_session_pool(df: pd.DataFrame) -> None:
    """Accumulates attacks generated on the Generate Attacks page into this browser session's
    state, so the Live Defense Demo and Closed Loop pages can act on exactly what this user
    produced - st.session_state is per-session (per browser tab), so this never mixes one
    visitor's generated attacks with another's."""
    existing = st.session_state.get(SESSION_POOL_KEY)
    st.session_state[SESSION_POOL_KEY] = pd.concat([existing, df], ignore_index=True) if existing is not None else df.copy()


def get_session_pool() -> pd.DataFrame:
    return st.session_state.get(SESSION_POOL_KEY, pd.DataFrame())


def clear_session_pool() -> None:
    st.session_state.pop(SESSION_POOL_KEY, None)


@st.cache_resource
def load_demo_backbone():
    """A small, committed sample (built by `scripts/build_demo_backbone.py`) rather than the
    full 493MB raw PaySim CSV - deploying this app (e.g. Streamlit Community Cloud) only has
    whatever's in the git repo, and the raw CSV can't live there (Kaggle terms + repo size), so
    the app's live-generation pages run off this small pre-extracted sample of the same real
    statistics instead."""
    import json

    import numpy as np
    import pandas as pd

    from src.generate.simulate import Backbone
    from src.generate.rule_based_agents import AgentContext

    legit_df = pd.read_parquet(PROCESSED_DIR / "demo_legit_sample.parquet")
    victim_balances = pd.read_parquet(PROCESSED_DIR / "demo_victim_balances.parquet")
    with open(PROCESSED_DIR / "demo_backbone_meta.json") as f:
        meta = json.load(f)

    rng = np.random.default_rng(123)
    ctx = AgentContext(
        rng=rng,
        real_accounts=legit_df["nameOrig"].astype(str).unique(),
        amount_quantiles=np.array(meta["amount_quantiles"]),
        max_step=meta["max_step"],
        victim_balances=victim_balances,
    )
    return Backbone(legit_df, legit_df.iloc[0:0], ctx, meta["tabular_attack_types"], meta["graph_attack_types"])
