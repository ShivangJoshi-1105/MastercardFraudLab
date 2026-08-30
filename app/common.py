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


@st.cache_resource
def load_demo_backbone():
    """A lighter-weight AgentContext (reads only the first 200k PaySim rows) for the app's live
    'generate N more incidents' button - full statistical fidelity isn't needed for a UI demo,
    only enough real signal to keep amounts/victims realistic, and this keeps the page snappy."""
    from src.generate.simulate import RAW_PATH, Backbone, DTYPES
    from src.generate.rule_based_agents import AgentContext, all_agents
    import numpy as np
    import pandas as pd

    df = pd.read_csv(RAW_PATH, dtype=DTYPES, nrows=200_000)
    legit_df = df[df["isFraud"] == 0].reset_index(drop=True)
    rng = np.random.default_rng(123)
    amount_quantiles = np.quantile(df["amount"].to_numpy(dtype=float), np.linspace(0, 1, 200))
    victim_pool = legit_df[legit_df["oldbalanceOrg"] > 1000][["nameOrig", "oldbalanceOrg"]].sample(
        n=min(5000, (legit_df["oldbalanceOrg"] > 1000).sum()), random_state=123
    ).rename(columns={"nameOrig": "account", "oldbalanceOrg": "balance"}).reset_index(drop=True)

    ctx = AgentContext(
        rng=rng,
        real_accounts=legit_df["nameOrig"].astype(str).unique(),
        amount_quantiles=amount_quantiles,
        max_step=int(df["step"].max()),
        victim_balances=victim_pool,
    )
    registry = all_agents()
    tabular_types = [name for name, cls in registry.items() if not cls.is_graph]
    graph_types = [name for name, cls in registry.items() if cls.is_graph]
    return Backbone(legit_df, df[df["isFraud"] == 1], ctx, tabular_types, graph_types)
