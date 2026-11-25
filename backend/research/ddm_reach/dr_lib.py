"""Statistics (atlas/P5-B definitions) and helpers for the DDM reachable-region study (plan v0.6, section 5)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dr_env as E  # noqa: E402,F401
from ddm_regime_atlas import full_characterize  # noqa: E402  (function used for the atlas and the mapping)
from s2_lib import vr_win  # noqa: E402  (P5-B stage 2 within-window overlapping variance ratio)


def robust_tail(r):
    """Q = (q99.5 - q0.5) / (q75 - q25) of the returns (plan section 5: robust companion of the kurtosis; defined before any result)."""
    q = np.quantile(r, [0.005, 0.25, 0.75, 0.995])
    iqr = q[2] - q[1]
    return float((q[3] - q[0]) / iqr) if iqr > 0 else float("nan")


def window_stats(closes):
    """closes: n+1 closes -> statistics of the n log returns, atlas definitions (data.characteristics via full_characterize)."""
    closes = np.asarray(closes, float)
    r = np.diff(np.log(closes))
    a = full_characterize(pd.DataFrame({"close": closes}))
    out = {k: float(a[k]) for k in ("std", "skewness", "kurtosis", "acf_1", "acf_5", "acf_20", "volclust_50")}
    out["skew"] = out.pop("skewness")
    out["vr20"] = vr_win(r, 20)
    out["vr60"] = vr_win(r, 60)
    out["Q"] = robust_tail(r)
    out["zero_frac"] = float(np.mean(r == 0.0))
    return out


def agg_stats(closes, a):
    """aggregation profile: the same window re-aggregated by a base bars (closes[::a])."""
    c = np.asarray(closes, float)[::a]
    s = full_characterize(pd.DataFrame({"close": c}))
    return {"acf_1": float(s["acf_1"]), "kurtosis": float(s["kurtosis"])}


def ticks_to_closes(ticks, tpc):
    return ticks[::tpc].astype(np.float64) * E.TICK


def windows(closes, L):
    """non-overlapping windows of L returns (L+1 closes, stride L)."""
    n = (len(closes) - 1) // L
    return [closes[k * L:k * L + L + 1] for k in range(n)]
