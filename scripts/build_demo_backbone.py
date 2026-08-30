"""
Builds a small, committable "demo backbone" sample (a few thousand rows + summary statistics)
from the real PaySim data, so the Streamlit app's live-generation pages (Generate Attacks, Fraud
Network Explorer) don't need the full 493MB raw CSV at deploy time.

Why this matters: the raw PaySim CSV can't be committed to GitHub (Kaggle terms + it's larger
than sensible for a code repo), and a platform like Streamlit Community Cloud only has whatever
the repo contains - it can't re-download a Kaggle dataset without secrets configured. Without
this step, the deployed app would work for the pages that only read `models/` and
`data/processed/test_set.parquet` (both committed) but crash on the two pages that need a live
AgentContext. This script produces the small artifact those two pages actually need instead.

Run once (after `scripts/download_data.py`), before deploying: it does not need retraining
anything, just a fresh read + sample of the real backbone statistics.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generate.simulate import load_backbone

OUT_DIR = ROOT / "data" / "processed"


def main():
    backbone = load_backbone(legit_sample_n=5000, seed=7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    backbone.legit_df.to_parquet(OUT_DIR / "demo_legit_sample.parquet")
    backbone.ctx.victim_balances.to_parquet(OUT_DIR / "demo_victim_balances.parquet")

    with open(OUT_DIR / "demo_backbone_meta.json", "w") as f:
        json.dump(
            {
                "amount_quantiles": backbone.ctx.amount_quantiles.tolist(),
                "max_step": backbone.ctx.max_step,
                "tabular_attack_types": backbone.tabular_attack_types,
                "graph_attack_types": backbone.graph_attack_types,
            },
            f,
        )

    print(f"Saved demo backbone sample: {len(backbone.legit_df)} legit rows, "
          f"{len(backbone.ctx.victim_balances)} victim candidates, max_step={backbone.ctx.max_step}")


if __name__ == "__main__":
    main()
