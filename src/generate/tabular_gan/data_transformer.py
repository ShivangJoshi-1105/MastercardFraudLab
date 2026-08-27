"""
Turns raw transaction rows into a fixed-width numeric tensor a GAN can train on, and back again.

**Why this file needs to exist at all** (the thing most from-scratch tabular GAN attempts get
wrong): naively feeding raw columns into a GAN — z-scoring `amount` and one-hot encoding `type` —
looks reasonable but fails in practice, because transaction amounts are heavily **multimodal**:
a "typical PAYMENT" cluster around a few hundred, a "typical CASH_OUT" cluster around a few
thousand, and a long tail of large transfers. A single global mean/std squashes all of that into
one Gaussian assumption, and a vanilla generator trained against it tends to mode-collapse onto
the single biggest cluster (you'll see this failure mode directly in `train.py`'s first
diagnostic run before the fix is applied here).

**The fix — mode-specific normalization** (the core idea behind CTGAN, simplified here so it's
readable and hand-written rather than imported): for each continuous column, fit a small
Gaussian-mixture model to discover its modes first. Every value is then represented as *which*
mode it belongs to (a one-hot vector) plus *where within that mode* it falls (a single scalar,
normalized by that mode's own mean/std). A generator producing "mode 2, +0.3 std" is a much
easier learning target than "47,382.19 rupees" directly, and it naturally reproduces multimodal
distributions instead of averaging them away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

CONTINUOUS_COLUMNS = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
DISCRETE_COLUMNS = ["type"]
N_MODES = 5
MODE_WEIGHT_THRESHOLD = 0.01  # drop modes the mixture barely uses


@dataclass
class ContinuousColumnInfo:
    name: str
    means: np.ndarray  # per active mode
    stds: np.ndarray
    n_modes: int
    start: int  # column offset in the transformed tensor
    width: int  # n_modes (one-hot) + 1 (scalar)


@dataclass
class DiscreteColumnInfo:
    name: str
    categories: list
    start: int
    width: int


class TabularDataTransformer:
    """Fit on a real DataFrame slice, then transform/inverse_transform between that DataFrame
    representation and the fixed-width tensor the generator/discriminator operate on."""

    def __init__(self):
        self.continuous_info: list[ContinuousColumnInfo] = []
        self.discrete_info: list[DiscreteColumnInfo] = []
        self.output_dim = 0
        self._gmms: dict[str, BayesianGaussianMixture] = {}

    def fit(self, df: pd.DataFrame) -> "TabularDataTransformer":
        offset = 0
        for col in CONTINUOUS_COLUMNS:
            values = df[col].to_numpy(dtype=np.float64).reshape(-1, 1)
            gmm = BayesianGaussianMixture(
                n_components=N_MODES,
                weight_concentration_prior=1e-3,  # encourages pruning unused modes
                max_iter=100,
                random_state=0,
            )
            gmm.fit(values)
            active = gmm.weights_ > MODE_WEIGHT_THRESHOLD
            n_active = max(int(active.sum()), 1)
            means = gmm.means_.flatten()[active][:n_active]
            stds = np.sqrt(gmm.covariances_.flatten()[active][:n_active]) + 1e-6
            self._gmms[col] = gmm
            width = n_active + 1
            self.continuous_info.append(ContinuousColumnInfo(col, means, stds, n_active, offset, width))
            offset += width

        for col in DISCRETE_COLUMNS:
            categories = sorted(df[col].astype(str).unique().tolist())
            width = len(categories)
            self.discrete_info.append(DiscreteColumnInfo(col, categories, offset, width))
            offset += width

        self.output_dim = offset
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        out = np.zeros((n, self.output_dim), dtype=np.float32)

        for info in self.continuous_info:
            values = df[info.name].to_numpy(dtype=np.float64).reshape(-1, 1)
            # assign each value to its nearest active mode (argmax density) rather than the full
            # CTGAN sampling procedure - simpler to invert deterministically, same core idea.
            dists = np.abs(values - info.means.reshape(1, -1)) / info.stds.reshape(1, -1)
            mode_idx = np.argmin(dists, axis=1)
            scalar = (values.flatten() - info.means[mode_idx]) / (4 * info.stds[mode_idx])
            scalar = np.clip(scalar, -1.0, 1.0)
            onehot = np.zeros((n, info.n_modes), dtype=np.float32)
            onehot[np.arange(n), mode_idx] = 1.0
            out[:, info.start : info.start + info.n_modes] = onehot
            out[:, info.start + info.n_modes] = scalar

        for info in self.discrete_info:
            cat_to_idx = {c: i for i, c in enumerate(info.categories)}
            idx = df[info.name].astype(str).map(cat_to_idx).fillna(0).to_numpy(dtype=int)
            onehot = np.zeros((n, info.width), dtype=np.float32)
            onehot[np.arange(n), idx] = 1.0
            out[:, info.start : info.start + info.width] = onehot

        return out

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        n = arr.shape[0]
        data = {}

        for info in self.continuous_info:
            block = arr[:, info.start : info.start + info.width]
            onehot, scalar = block[:, : info.n_modes], block[:, info.n_modes]
            mode_idx = np.argmax(onehot, axis=1)
            scalar = np.clip(scalar, -1.0, 1.0)
            values = scalar * 4 * info.stds[mode_idx] + info.means[mode_idx]
            if info.name != "step":  # balances/amounts can't be negative
                values = np.clip(values, 0, None)
            data[info.name] = values

        for info in self.discrete_info:
            block = arr[:, info.start : info.start + info.width]
            idx = np.argmax(block, axis=1)
            data[info.name] = [info.categories[i] for i in idx]

        return pd.DataFrame(data)

    def activation_spec(self) -> list[tuple[int, int, str]]:
        """(start, width, activation_kind) per block, so the generator knows which activation to
        apply where: 'softmax' for one-hot mode/category blocks, 'tanh' for the continuous
        within-mode scalar. This is why the generator can't just end in one global activation."""
        spec = []
        for info in self.continuous_info:
            spec.append((info.start, info.n_modes, "softmax"))
            spec.append((info.start + info.n_modes, 1, "tanh"))
        for info in self.discrete_info:
            spec.append((info.start, info.width, "softmax"))
        return spec
