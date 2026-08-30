"""Loads the structured attack taxonomy for the Streamlit app's taxonomy explorer page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TAXONOMY_PATH = Path(__file__).resolve().parent / "attack_taxonomy.json"


def load_taxonomy() -> pd.DataFrame:
    with open(TAXONOMY_PATH) as f:
        data = json.load(f)
    return pd.DataFrame(data)
