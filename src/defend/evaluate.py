"""
Metric reporting for the defense classifier. Reports the standard imbalanced-classification set
(precision/recall/F1/ROC-AUC/PR-AUC) plus one the brief calls out by name: the **false-positive
rate specifically on legitimate transactions** ("keeping false positives on legitimate payments
low"). That number is easy to bury inside an aggregate confusion matrix, so it gets its own
explicit field here rather than requiring the reader to compute it from a matrix.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "false_positive_rate_on_legit": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_legit": int(tn + fp),
        "n_fraud": int(fn + tp),
    }


def false_negatives_mask(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    y_pred = (y_prob >= threshold).astype(int)
    return (y_true == 1) & (y_pred == 0)
