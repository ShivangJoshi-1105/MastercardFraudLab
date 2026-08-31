"""
Builds a small, committable, stratified sample of the training set specifically for the
Streamlit app's live "run a closed-loop iteration now" button (see app/pages/5_Closed_Loop.py).

The full train_set.parquet (242,720 rows) is deliberately not committed to the repo (see
.gitignore) - it's large, and the live demo button needs to complete in well under a minute on a
shared/free-tier host, not minutes. This script draws a smaller stratified sample (same label
balance) that keeps the surrogate distillation and retrain step fast enough for an interactive
button click, at the cost of using less data than the full offline pipeline run documented in
README.md and the walkthrough - a deliberate, documented trade-off between two different use
cases (interactive demo vs. full offline evaluation), not an inconsistency.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLE_SIZE_PER_CLASS = 3000


def main():
    train_df = pd.read_parquet(PROCESSED_DIR / "train_set.parquet")
    legit = train_df[train_df["label"] == 0].sample(n=min(SAMPLE_SIZE_PER_CLASS, (train_df["label"] == 0).sum()), random_state=11)
    fraud = train_df[train_df["label"] == 1].sample(n=min(SAMPLE_SIZE_PER_CLASS, (train_df["label"] == 1).sum()), random_state=11)
    sample = pd.concat([legit, fraud], ignore_index=True)
    sample.to_parquet(PROCESSED_DIR / "closed_loop_demo_train_sample.parquet")
    print(f"Saved {len(sample)} rows ({len(legit)} legit, {len(fraud)} fraud) for the live closed-loop demo.")


if __name__ == "__main__":
    main()
