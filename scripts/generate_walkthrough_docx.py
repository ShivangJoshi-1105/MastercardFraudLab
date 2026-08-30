"""
Generates docs/WALKTHROUGH.docx - the required Word write-up - by pulling real numbers out of
the artifacts `run_pipeline.py` produced (metrics.json, fidelity reports, closed_loop.json)
rather than hand-typing results that could drift out of sync with the actual trained model.

Run this *after* `scripts/run_pipeline.py` has completed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
REPORTS_DIR = MODELS_DIR / "reports"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def add_heading(doc, text, level=1):
    doc.add_heading(text, level=level)


def add_body(doc, text):
    p = doc.add_paragraph(text)
    p.style.font.size = Pt(11)
    return p


def main():
    from src.identify.taxonomy import load_taxonomy

    metrics = load_json(MODELS_DIR / "defense" / "metrics.json")
    tab_fidelity = load_json(REPORTS_DIR / "tabular_fidelity.json")
    graph_fidelity = load_json(REPORTS_DIR / "graph_fidelity.json")
    closed_loop = load_json(REPORTS_DIR / "closed_loop.json")
    taxonomy = load_taxonomy()

    doc = Document()
    doc.add_heading("AI Defense Lab for Payment Security", level=0)
    doc.add_paragraph("Mastercard Innovation Challenge @ GFF 2026 — Solution Walkthrough")

    # ---------------------------------------------------------------- Section 1
    add_heading(doc, "1. Novel fraud attacks identified")
    add_body(
        doc,
        f"We built a taxonomy of {len(taxonomy)} distinct GenAI-powered payment fraud vectors, "
        f"organized across {taxonomy['category'].nunique()} categories spanning the full payment "
        "lifecycle: identity/onboarding, account takeover, card-not-present, money-mule networks, "
        "social engineering/authorized push payment, systemic attacks on the defense itself, and "
        "emerging agentic-commerce risk. Each vector is grounded in a specific GenAI capability "
        "(voice cloning, deepfake video, LLM-driven personalization/automation, agentic bots) and "
        "mapped to the transactional fingerprint it leaves behind — the behavioral trace a "
        f"payments pipeline can actually observe. {int(taxonomy['simulated'].sum())} of these "
        "vectors are concretely simulated in code; the remainder are documented for completeness "
        "since their signal lives outside transaction data (e.g. document authenticity, security "
        "properties of internal fraud-ops tooling).",
    )
    add_body(doc, "Key vectors simulated in code:")
    for _, row in taxonomy[taxonomy["simulated"]].iterrows():
        doc.add_paragraph(f"{row['id']} — {row['name']}: {row['mechanism']}", style="List Bullet")

    # ---------------------------------------------------------------- Section 2
    add_heading(doc, "2. How the system generates and simulates these attacks")
    add_body(
        doc,
        "Generation happens in two layers. First, 10 rule-based attack agents (7 behavioral, 3 "
        "network/graph) inject precisely-labeled fraud incidents onto a real transaction backbone "
        "(the PaySim mobile-money dataset), encoding each attack type's exact transactional "
        "fingerprint from the taxonomy. Second, two from-scratch generative models learn to scale "
        "those patterns up:",
    )
    doc.add_paragraph(
        "A tabular WGAN-GP (Wasserstein GAN with gradient penalty) for row-level behavioral fraud "
        "(card-testing, structuring, ATO bursts, synthetic-identity bust-out, BEC and romance-scam "
        "proxies), using mode-specific normalization to handle the multimodal amount distributions "
        "that cause vanilla GAN training to mode-collapse.",
        style="List Bullet",
    )
    doc.add_paragraph(
        "A graph GAN (hand-rolled graph-convolution generator and discriminator, no external "
        "graph-ML library) for network-structured fraud — mule layering chains, collusive rings, "
        "and coordinated fan-in mule bursts — since no single transaction in a laundering ring "
        "looks unusual in isolation; only the topology does.",
        style="List Bullet",
    )
    doc.add_paragraph(
        "A 'Red-Team GAN' — our own modification to the GAN objective — adds a second loss term "
        "to the generator: evading a distilled surrogate of the live defense classifier, not just "
        "fooling the critic. This is what closes the loop (Section 4).",
        style="List Bullet",
    )
    add_body(
        doc,
        f"Fidelity results: the tabular GAN's synthetic fraud achieves a discriminative AUC of "
        f"{tab_fidelity['discriminative_auc']:.3f} against real rule-based fraud (0.5 = a classifier "
        f"cannot tell real from synthetic apart) with a correlation-matrix difference of "
        f"{tab_fidelity['correlation_diff']:.3f}. The graph GAN's synthetic mule/ring topologies "
        f"show a degree-distribution KS statistic of {graph_fidelity['degree_ks']:.3f} against real "
        f"topologies, with real vs. synthetic cycle rates of "
        f"{graph_fidelity['real_cycle_rate']:.2f} vs. {graph_fidelity['synth_cycle_rate']:.2f} "
        "(collusive rings are, by definition, cycles — this is the key structural check).",
    )

    # ---------------------------------------------------------------- Section 3
    add_heading(doc, "3. Detection and mitigation model, with efficacy results")
    add_body(
        doc,
        "The defense is an XGBoost classifier on engineered features: balance-consistency "
        "residuals, trailing 24-hour transaction velocity/sum per account, and per-account "
        "recency (transactions-so-far), trained on real legitimate transactions plus rule-based "
        "and GAN-augmented synthetic fraud, with a group-aware train/test split (whole incidents "
        "and accounts, never individual rows, are held out) to avoid leakage.",
    )
    m = metrics
    doc.add_paragraph(
        f"Precision: {m['precision']:.3f}  |  Recall: {m['recall']:.3f}  |  F1: {m['f1']:.3f}  |  "
        f"ROC-AUC: {m['roc_auc']:.3f}  |  PR-AUC: {m['pr_auc']:.3f}",
    )
    doc.add_paragraph(
        f"False-positive rate on legitimate transactions: {m['false_positive_rate_on_legit']:.4f} "
        f"({m['confusion_matrix']['fp']} false positives out of {m['n_legit']} legitimate "
        "transactions in the held-out set) — kept explicitly low per the brief's requirement.",
    )
    add_body(
        doc,
        "SHAP (TreeExplainer) provides per-prediction feature attributions in the live demo, so "
        "any flagged transaction can be explained in terms of which engineered feature drove the "
        "score — necessary for analyst trust and regulatory explainability in a real deployment.",
    )

    # ---------------------------------------------------------------- Section 4
    add_heading(doc, "4. Closing the loop, and real-world feasibility")
    add_body(
        doc,
        f"One closed-loop iteration: the defense's false negatives on the held-out set identified "
        f"'{closed_loop['target_attack_type']}' as its weakest attack type. The Red-Team GAN was "
        "trained against a surrogate of the live classifier, conditioned on that type, producing "
        "a harder synthetic batch explicitly optimized to evade the current defense. Folding that "
        "batch in and retraining moved recall from "
        f"{closed_loop['metrics_before']['recall']:.3f} to "
        f"{closed_loop['metrics_after']['recall']:.3f}, while false-positive rate on legit "
        f"transactions changed from {closed_loop['metrics_before']['false_positive_rate_on_legit']:.4f} "
        f"to {closed_loop['metrics_after']['false_positive_rate_on_legit']:.4f} — a concrete, "
        "measured demonstration that the attacker and defender genuinely adapt to each other, "
        "rather than three independent pillars presented side by side.",
    )
    add_body(
        doc,
        "Real-world feasibility: every simulated attack is grounded in its observable transactional "
        "fingerprint rather than assuming a payments system can directly perceive a deepfake or a "
        "cloned voice — this is deliberate, since a production fraud system genuinely can only act "
        "on transaction-level signal. The velocity/graph features generalize to any transaction "
        "stream with account, amount, timestamp, and counterparty fields (not PaySim-specific), and "
        "the closed-loop retraining pattern maps directly onto a real fraud team's periodic model "
        "refresh cycle, with the Red-Team GAN standing in for the adversarial stress-testing a bank "
        "would otherwise have to commission manually or wait to observe in production losses.",
    )

    out_path = ROOT / "docs" / "WALKTHROUGH.docx"
    doc.save(str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
