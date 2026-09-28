"""Paired bootstrap over whole groups (days) -- windows within a day are strongly dependent."""
from __future__ import annotations

import numpy as np


def paired_block_bootstrap_ci(loss_a: np.ndarray, loss_b: np.ndarray, groups: np.ndarray,
                              n_boot: int = 500, level: float = 0.95, seed: int = 1) -> tuple[float, float]:
    """CI of mean(loss_a - loss_b), resampling whole groups with replacement. Positive = b better."""
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    uniq, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=d)
    c = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    a = (1 - level) / 2
    return float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))
