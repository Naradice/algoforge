"""Future-only targets. Realized volatility is always the RMS of log returns -- never the std,
which removes the mean (mixing the two once produced a wrong HAR bar; see
docs/research-periodic-forcing-transfer.md, Phase 8c)."""
from __future__ import annotations

import numpy as np

EPS = 1e-7


def rms(a: np.ndarray) -> np.ndarray:
    return np.sqrt((a ** 2).mean(axis=1))


def future_log_rv(X: np.ndarray, F: np.ndarray, h: int) -> np.ndarray:
    """log RMS of the next h returns."""
    return np.log(rms(F[:, :h]) + EPS)


def vol_change(X: np.ndarray, F: np.ndarray, h: int) -> np.ndarray:
    """log RMS of the next h returns minus log RMS of the last h inputs."""
    return np.log(rms(F[:, :h]) + EPS) - np.log(rms(X[:, -h:]) + EPS)


def jump(X: np.ndarray, F: np.ndarray, h: int, k: float = 4.0) -> np.ndarray:
    """1 if any |r| in the next h returns exceeds k x the RMS of the whole input window."""
    return (np.abs(F[:, :h]).max(axis=1) > k * (rms(X) + EPS)).astype(np.int8)


TARGETS = {"future_log_rv": future_log_rv, "vol_change": vol_change, "jump": jump}
CLASSIFICATION = {"jump"}


def make_target(kind: str, X: np.ndarray, F: np.ndarray, h: int, **kw) -> np.ndarray:
    if kind not in TARGETS:
        raise ValueError(f"unknown target {kind!r}; expected one of {sorted(TARGETS)}")
    if h > F.shape[1]:
        raise ValueError(f"horizon {h} exceeds the windows' horizon {F.shape[1]}")
    return TARGETS[kind](X, F, h, **kw)
