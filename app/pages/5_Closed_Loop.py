import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import artifacts_ready, load_json, REPORTS_DIR

st.set_page_config(page_title="Closed Loop", page_icon="🔁", layout="wide")
st.title("🔁 Closing the Loop — Red-Team GAN")
st.caption("Our own GAN modification: the generator is rewarded for evading the *current* defense, not just for fooling the critic")

if not artifacts_ready():
    st.warning("Run `python scripts/run_pipeline.py` first.")
    st.stop()

result = load_json(REPORTS_DIR / "closed_loop.json")

st.markdown(
    f"""
**How this iteration worked:**
1. The current defense's false negatives on the held-out test set were mined to find its weakest
   attack type: **`{result['target_attack_type']}`**.
2. A differentiable surrogate of the live XGBoost classifier was distilled in the tabular GAN's
   representation space.
3. The Red-Team GAN generator was trained against that surrogate, conditioned on
   `{result['target_attack_type']}`, so its output is explicitly optimized to be scored as
   *legitimate* by the current defense.
4. The harder synthetic batch was folded into the training set and the classifier was retrained
   from scratch, then re-evaluated on the exact same held-out test set as before.
"""
)

before, after = result["metrics_before"], result["metrics_after"]
metric_names = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
comparison = pd.DataFrame(
    {
        "metric": metric_names * 2,
        "value": [before[m] for m in metric_names] + [after[m] for m in metric_names],
        "when": ["before"] * len(metric_names) + ["after"] * len(metric_names),
    }
)
fig = px.bar(comparison, x="metric", y="value", color="when", barmode="group", title="Defense performance before vs. after the closed-loop iteration")
st.plotly_chart(fig, use_container_width=True)

cols = st.columns(3)
cols[0].metric("Recall - before", f"{before['recall']:.3f}")
cols[1].metric("Recall - after", f"{after['recall']:.3f}", delta=f"{after['recall'] - before['recall']:+.3f}")
cols[2].metric("FPR on legit - after", f"{after['false_positive_rate_on_legit']:.4f}", delta=f"{after['false_positive_rate_on_legit'] - before['false_positive_rate_on_legit']:+.4f}", delta_color="inverse")

if "redteam_gan_history" in result:
    st.divider()
    st.subheader("Red-Team GAN training curves")
    rt = result["redteam_gan_history"]
    rt_df = pd.DataFrame({"epoch": range(len(rt["evasion_loss"])), "evasion_loss": rt["evasion_loss"], "critic_loss": rt["critic_loss"]})
    st.line_chart(rt_df.set_index("epoch"))
    st.caption("Falling evasion loss means the generator is getting better at producing fraud the surrogate defense scores as legitimate.")

history_path = REPORTS_DIR / "gan_training_history.json"
if history_path.exists():
    history = load_json(history_path)
    st.divider()
    st.subheader("Tabular WGAN-GP training curves (base generator)")
    hist_df = pd.DataFrame({"epoch": range(len(history["tabular"]["critic_loss"])), "critic_loss": history["tabular"]["critic_loss"], "gen_loss": history["tabular"]["gen_loss"]})
    st.line_chart(hist_df.set_index("epoch"))

st.info(
    "This is one closed-loop iteration. Running it again mines the *new* weakest spot and repeats "
    "the cycle — the generator and the defense keep adapting to each other, which is the core "
    "'red-team/blue-team' framing the challenge brief asks for."
)
