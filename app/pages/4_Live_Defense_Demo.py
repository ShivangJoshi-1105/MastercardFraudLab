import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import artifacts_ready, get_session_pool, load_defense_model, load_json, load_parquet, MODELS_DIR
from src.defend.evaluate import evaluate_predictions
from src.defend.features import FEATURE_COLUMNS, engineer_features, get_feature_matrix

st.set_page_config(page_title="Live Defense Demo", layout="wide")
st.title("Pillar 3 — Live Defense Demo")
st.caption("XGBoost on engineered velocity/balance/graph features, evaluated on a held-out set the model has never seen")

if not artifacts_ready():
    st.warning("Run `python scripts/run_pipeline.py` first.")
    st.stop()

model = load_defense_model()
test_df = load_parquet("test_set.parquet")
saved_metrics = load_json(MODELS_DIR / "defense" / "metrics.json")

session_pool = get_session_pool()
st.subheader("Score what you generated")
if len(session_pool) == 0:
    st.info("Your session pool is empty — generate some attacks on the **Generate Attacks** page first, then come back here to see whether the current defense catches them.")
else:
    session_engineered = engineer_features(session_pool)
    session_prob = model.predict_proba(get_feature_matrix(session_engineered))[:, 1]
    caught = int((session_prob >= 0.5).sum())
    st.metric(f"Caught by the current live model (threshold 0.5)", f"{caught} / {len(session_pool)}")
    display = session_engineered.assign(fraud_probability=session_prob, flagged=session_prob >= 0.5)
    st.dataframe(
        display[["step", "type", "amount", "attack_type", "fraud_probability", "flagged"]].sort_values("fraud_probability", ascending=False),
        use_container_width=True, height=250,
    )
    st.caption("This is pure inference against the already-trained model — nothing here retrains anything. To actually retrain on these, use the Closed Loop page.")

st.divider()
X_test = get_feature_matrix(test_df)
y_test = test_df["label"].to_numpy()
y_prob = model.predict_proba(X_test)[:, 1]

st.subheader("Held-out test set performance")
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
metrics = evaluate_predictions(y_test, y_prob, threshold)

cols = st.columns(6)
cols[0].metric("Precision", f"{metrics['precision']:.3f}")
cols[1].metric("Recall", f"{metrics['recall']:.3f}")
cols[2].metric("F1", f"{metrics['f1']:.3f}")
cols[3].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
cols[4].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
cols[5].metric("FPR on legit", f"{metrics['false_positive_rate_on_legit']:.4f}", help="False positives / all legitimate transactions — kept explicitly low per the brief")

cm = metrics["confusion_matrix"]
st.markdown(f"**Confusion matrix** — TN: {cm['tn']:,} · FP: {cm['fp']:,} · FN: {cm['fn']:,} · TP: {cm['tp']:,}")

st.divider()
st.subheader("Live transaction stream (sampled from the held-out set)")
sample_n = st.slider("Stream size", 20, 500, 100)
sample = test_df.sample(n=min(sample_n, len(test_df)), random_state=np.random.randint(0, 10_000)).reset_index(drop=True)
sample_prob = model.predict_proba(get_feature_matrix(sample))[:, 1]
sample = sample.assign(fraud_probability=sample_prob, flagged=sample_prob >= threshold)

display_cols = ["step", "type", "amount", "attack_type", "label", "fraud_probability", "flagged"]
st.dataframe(
    sample[display_cols].sort_values("fraud_probability", ascending=False).style.apply(
        lambda row: ["background-color: #ffcccc" if row["flagged"] else "" for _ in row], axis=1
    ),
    use_container_width=True,
    height=350,
)

st.divider()
st.subheader("Why did the model flag this one? (SHAP)")
flagged_rows = sample[sample["flagged"]].reset_index(drop=True)
if len(flagged_rows) == 0:
    st.info("No transactions flagged at this threshold in the current sample.")
else:
    row_idx = st.selectbox("Pick a flagged transaction", flagged_rows.index, format_func=lambda i: f"{flagged_rows.loc[i, 'attack_type']} — amount {flagged_rows.loc[i, 'amount']:.2f} — p={flagged_rows.loc[i, 'fraud_probability']:.3f}")
    explainer = shap.TreeExplainer(model)
    row_features = get_feature_matrix(flagged_rows.iloc[[row_idx]])
    shap_values = explainer(row_features)

    contrib = pd.DataFrame({"feature": FEATURE_COLUMNS, "shap_value": shap_values.values[0]}).sort_values("shap_value", key=abs, ascending=False).head(10)
    fig = px.bar(contrib, x="shap_value", y="feature", orientation="h", title="Top feature contributions to this prediction", color="shap_value", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)
