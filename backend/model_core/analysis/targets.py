"""Future-only targets. Realized volatility is always the RMS of log returns -- never the std,
which removes the mean (mixing the two once produced a wrong HAR bar; see
docs/research-periodic-forcing-transfer.md, Phase 8c).

A target may return NaN for windows where it is undefined (e.g. a flat future for `direction`);
assess_target drops those windows before splitting."""
from __future__ import annotations

import numpy as np

EPS = 1e-7
FLAT = 1e-9   # |sum of log returns| below this is "no move" (price ticks are ~1e-5 in log terms)


def rms(a: np.ndarray) -> np.ndarray:
    return np.sqrt((a ** 2).mean(axis=1))


def efficiency_ratio(a: np.ndarray) -> np.ndarray:
    """|net move| / path length per row, in [0, 1]: 1 = straight trend, ~0 = range. NaN if no move."""
    path = np.abs(a).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(path > FLAT, np.abs(a.sum(axis=1)) / path, np.nan)


def future_log_rv(X: np.ndarray, F: np.ndarray, h: int) -> np.ndarray:
    """log RMS of the next h returns."""
    return np.log(rms(F[:, :h]) + EPS)


def vol_change(X: np.ndarray, F: np.ndarray, h: int) -> np.ndarray:
    """log RMS of the next h returns minus log RMS of the last h inputs."""
    return np.log(rms(F[:, :h]) + EPS) - np.log(rms(X[:, -h:]) + EPS)


def jump(X: np.ndarray, F: np.ndarray, h: int, k: float = 4.0) -> np.ndarray:
    """1 if any |r| in the next h returns exceeds k x the RMS of the whole input window."""
    return (np.abs(F[:, :h]).max(axis=1) > k * (rms(X) + EPS)).astype(np.int8)


def future_return(X: np.ndarray, F: np.ndarray, h: int) -> np.ndarray:
    """Log return over the next h bars (sum of the next h log returns). Reference: 0 (random walk)."""
    return F[:, :h].sum(axis=1)


def direction(X: np.ndarray, F: np.ndarray, h: int) -> np.ndarray:
    """1 if the price is higher h bars ahead, 0 if lower, NaN if unchanged."""
    s = F[:, :h].sum(axis=1)
    return np.where(np.abs(s) < FLAT, np.nan, (s > 0).astype(np.float64))


def extreme(X: np.ndarray, F: np.ndarray, h: int, q: float = 0.999) -> np.ndarray:
    """1 if any |r| in the next h returns exceeds the q-quantile of |r| over the whole series --
    an absolute threshold, unlike `jump` (relative to the window's own RMS), so it mostly asks
    "will volatility be high" plus whatever tail structure is left beyond that."""
    thr = np.quantile(np.abs(X[:, -1]), q)
    return (np.abs(F[:, :h]).max(axis=1) > thr).astype(np.int8)


def trend_er(X: np.ndarray, F: np.ndarray, h: int) -> np.ndarray:
    """Efficiency ratio of the next h returns (trend vs range regime). Reference: the ER of the last h inputs."""
    return efficiency_ratio(F[:, :h])


def trend_change(X: np.ndarray, F: np.ndarray, h: int) -> np.ndarray:
    """Future ER minus the ER of the last h inputs: regime transition toward trend (>0) or range (<0)."""
    return efficiency_ratio(F[:, :h]) - efficiency_ratio(X[:, -h:])


TARGETS = {"future_log_rv": future_log_rv, "vol_change": vol_change, "jump": jump,
           "future_return": future_return, "direction": direction, "extreme": extreme,
           "trend_er": trend_er, "trend_change": trend_change}
CLASSIFICATION = {"jump", "direction", "extreme"}
# Targets that depend on the sign of returns: the baselines also get the raw lagged returns
# (an AR(obs) term), which vol-memory features alone do not carry.
SIGNED = {"future_return", "direction"}
# Targets needing the last h inputs to be defined (h <= obs).
NEEDS_INPUT_H = {"vol_change", "trend_change"}


def persistence(kind: str, X: np.ndarray, h: int) -> np.ndarray:
    """The naive "carry the recent past forward" forecast for a regression target."""
    h_in = min(h, X.shape[1])
    if kind == "future_log_rv":
        return np.log(rms(X[:, -h_in:]) + EPS)
    if kind == "trend_er":
        return np.nan_to_num(efficiency_ratio(X[:, -h_in:]), nan=0.0)
    return np.zeros(len(X))   # changes and returns: "no change" / random walk


def make_target(kind: str, X: np.ndarray, F: np.ndarray, h: int, **kw) -> np.ndarray:
    if kind not in TARGETS:
        raise ValueError(f"unknown target {kind!r}; expected one of {sorted(TARGETS)}")
    if h > F.shape[1]:
        raise ValueError(f"horizon {h} exceeds the windows' horizon {F.shape[1]}")
    if kind in NEEDS_INPUT_H and h > X.shape[1]:
        raise ValueError(f"target {kind!r} compares against the last {h} inputs; horizon must be <= obs")
    return TARGETS[kind](X, F, h, **kw)
