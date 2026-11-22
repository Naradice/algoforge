"""P5-B stage 2 library (plan v0.2): per-path statistics with the atlas/mapping definitions (data.characteristics via full_characterize), ACF profile,
within-path variance ratio VR_win. Cross-path quantities are computed in the analysis script."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s2_env as E  # noqa: E402,F401
from ddm_regime_atlas import full_characterize  # noqa: E402  (the function used for the atlas and the mapping)

LAGS = 60
KS = (5, 20, 60)
POS_BLOCKS = ((0, 100), (100, 200), (200, 1000), (1000, 1440))
HZ = (1, 4, 12, 24)


def acf_profile(x, nlags=LAGS):
    """sample autocorrelation (mean-subtracted, denominator sum of squares over all n; same estimator as statsmodels acf with adjusted=False)."""
    x = np.asarray(x, float)
    d = x - x.mean()
    den = float(np.dot(d, d))
    n = len(x)
    return np.array([np.dot(d[:n - j], d[j:]) / den for j in range(1, nlags + 1)])


def vr_win(r, k):
    """within-window variance ratio (overlapping sums of k returns, variance around their mean, over k times the variance of single returns)."""
    c = np.concatenate([[0.0], np.cumsum(r)])
    s = c[k:] - c[:-k]
    return float(s.var() / (k * r.var()))


def vr_win_brute(r, k):
    n = len(r)
    s = np.array([r[i:i + k].sum() for i in range(n - k + 1)])
    return float(np.mean((s - s.mean()) ** 2) / (k * np.mean((r - r.mean()) ** 2)))


def path_stats(closes):
    """closes: 1441 closes of a regenerated path (or 1001 closes of a real window)."""
    closes = np.asarray(closes, float)
    r = np.diff(np.log(closes))
    n1 = min(1000, len(r))
    r1 = r[:n1]
    out = {"atlas": full_characterize(pd.DataFrame({"close": closes[:n1 + 1]})), "var1000": float(r1.var()), "mean1000": float(r1.mean()),
           "acf": acf_profile(r1).tolist(), "vr_win": {str(k): vr_win(r1, k) for k in KS}}
    if len(r) >= 1440:
        out["var_full"] = float(r.var())
        out["var_60h"] = {str(h): float(r[:60 * h].var()) for h in HZ}
        out["pos_std"] = [float(r[a:b].std()) for a, b in POS_BLOCKS]
        out["kurt_full"] = float(sps.kurtosis(r, fisher=False))
        out["mean_full"] = float(r.mean())
    return out


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def closes_path(cond, q):
    return os.path.join(E.OUT, f"closes_{cond}_q{q}.npy")


def load_mapping():
    return json.load(open(os.path.join(E.RUN_DIR, "ddm_mapping.json")))
