"""
Answers the "Fidelity of attacks in simulation" judging criterion directly: how close is our
synthetic fraud to the real distributional shape of the thing it's imitating? Three
complementary metrics, because no single number is trustworthy alone:

1. **Per-feature KS statistic** — for each continuous column, the two-sample Kolmogorov-Smirnov
   test statistic between real and synthetic values (0 = identical distributions, 1 = totally
   disjoint). Cheap, interpretable, and per-column, so it points at exactly which feature is
   drifting if fidelity is poor.
2. **Discriminative score** — train a fresh classifier (logistic regression) whose only job is
   "is this row real or synthetic?" on a held-out mix, then report its test AUC. An AUC near 0.5
   means the classifier can't tell them apart (high fidelity); an AUC near 1.0 means synthetic
   rows are trivially distinguishable (low fidelity). This is the single most holistic fidelity
   metric here because it looks at the joint distribution, not one column at a time.
3. **Correlation-matrix difference** — the mean absolute difference between the real and
   synthetic correlation matrices over the continuous columns, catching cases where individual
   column distributions match but the *relationships between* columns don't (e.g. amount and
   balance no longer move together realistically).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def ks_statistics(real_df: pd.DataFrame, synth_df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return {col: float(ks_2samp(real_df[col], synth_df[col]).statistic) for col in columns}


def discriminative_score(real_df: pd.DataFrame, synth_df: pd.DataFrame, columns: list[str]) -> float:
    real = real_df[columns].to_numpy(dtype=np.float64)
    synth = synth_df[columns].to_numpy(dtype=np.float64)
    X = np.vstack([real, synth])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(synth))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
    probs = clf.predict_proba(scaler.transform(X_test))[:, 1]
    return float(roc_auc_score(y_test, probs))


def correlation_diff(real_df: pd.DataFrame, synth_df: pd.DataFrame, columns: list[str]) -> float:
    real_corr = real_df[columns].corr().to_numpy()
    synth_corr = synth_df[columns].corr().to_numpy()
    return float(np.nanmean(np.abs(real_corr - synth_corr)))


def fidelity_report(real_df: pd.DataFrame, synth_df: pd.DataFrame, columns: list[str]) -> dict:
    return {
        "ks_statistics": ks_statistics(real_df, synth_df, columns),
        "discriminative_auc": discriminative_score(real_df, synth_df, columns),
        "correlation_diff": correlation_diff(real_df, synth_df, columns),
    }
