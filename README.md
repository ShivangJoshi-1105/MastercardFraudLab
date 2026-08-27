# AI Defense Lab for Payment Security

Mastercard Innovation Challenge @ GFF 2026 — a closed-loop red-team/blue-team system that
identifies GenAI-powered payment fraud, simulates it at scale with high fidelity, and defends
against it.

## The closed loop

```
IDENTIFY  →  GENERATE  →  DEFEND
   ▲                          │
   └──────── feedback ────────┘
```

1. **Identify** (`docs/ATTACK_TAXONOMY.md`) — 25 GenAI-powered payment fraud vectors across
   identity/onboarding, account takeover, card-not-present, money-mule networks, social
   engineering, systemic/adversarial, and emerging agentic-commerce risk. 10 of them are
   concretely simulated in code.
2. **Generate** (`src/generate/`) — 10 rule-based attack agents (7 behavioral/tabular, 3
   network/graph) inject precisely-labeled fraud into a real transaction backbone
   ([PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)), then two from-scratch generative
   models learn and scale up those patterns:
   - A **tabular WGAN-GP** for row-level behavioral fraud (card-testing, structuring, ATO
     bursts, synthetic-identity bust-out, BEC/romance-scam proxies).
   - A **graph GAN** (hand-rolled GNN generator/discriminator, no external graph-ML library) for
     network-structured fraud (mule chains, collusive rings, fan-in bursts) — this is the "novel
     technique" the tabular GAN structurally cannot represent, since no single transaction in a
     laundering ring looks unusual on its own.
   - A **Red-Team GAN**: our own modification to the GAN objective — the generator is rewarded
     not just for fooling the discriminator, but for evading a live snapshot of the current
     defense classifier. This is what turns the pipeline into an actual closed loop.
3. **Defend** (`src/defend/`) — an XGBoost classifier on engineered velocity/recency/graph
   features, evaluated on precision/recall/F1/AUC-ROC/AUC-PR with explicit false-positive-rate
   reporting on legitimate transactions, plus SHAP explainability.
4. **Closed loop** (`src/closed_loop/`) — mines the classifier's false negatives, feeds them to
   the Red-Team GAN to generate a harder adversarial batch, retrains, and demonstrates the
   metric improvement.

All of this is demoed live in a 5-page Streamlit app (`app/`).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts\download_data.py       # fetches PaySim into data/raw/
python scripts\run_pipeline.py        # runs the full identify->generate->defend pipeline
streamlit run app\Home.py             # launches the interactive demo
```

## Repository layout

```
docs/ATTACK_TAXONOMY.md      Pillar 1 deliverable
src/identify/                 Structured taxonomy consumed by the app
src/generate/
  rule_based_agents/           10 attack agents (7 tabular + 3 graph)
  tabular_gan/                  From-scratch WGAN-GP for behavioral fraud
  graph_gan/                    From-scratch graph GAN for network fraud
  defense_aware_gan.py          Red-Team GAN (evasion-loss modification)
  fidelity_eval.py               Tabular fidelity metrics
src/defend/                    Feature engineering + XGBoost classifier + SHAP
src/closed_loop/               False-negative mining -> Red-Team GAN -> retrain
app/                            Streamlit prototype (5 pages)
scripts/                        download_data.py, run_pipeline.py
tests/                          Sanity tests for agents + features
```

## Why these design choices

See `docs/ATTACK_TAXONOMY.md` for the fraud taxonomy and the reasoning behind which vectors are
simulated vs. documented, and inline docstrings/comments in `src/generate/` for the GAN design
rationale (why WGAN-GP over vanilla GAN loss, why mode-specific normalization, why a hand-rolled
GNN, why the Red-Team GAN's evasion loss closes the loop).
