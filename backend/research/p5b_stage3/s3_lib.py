"""P5-B stage 3 library: VRproxy(60) (plan v0.3 section 3.1), valid atlas points, alternative selection CF1 (section 3.3)."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s3_env as E  # noqa: E402,F401
import s2_lib as S2  # noqa: E402,F401
import p5b_lib as L5  # noqa: E402,F401

J = np.arange(1, 60)


def vr_proxy(acf1, acf5, acf20):
    """VRproxy(60) = 1 + 2 sum_{j=1}^{59} (1 - j/60) rho_j; rho_j piecewise linear through (1, acf1), (5, acf5), (20, acf20); 0 for j >= 21. Vectorised over inputs."""
    acf1, acf5, acf20 = np.broadcast_arrays(np.asarray(acf1, float), np.asarray(acf5, float), np.asarray(acf20, float))
    shp = acf1.shape
    out = np.empty(acf1.size)
    w = 1 - J / 60
    for i, (a, b, c) in enumerate(zip(acf1.ravel(), acf5.ravel(), acf20.ravel())):
        rho = np.interp(J, [1, 5, 20], [a, b, c], right=0.0)
        out[i] = 1 + 2 * float(np.sum(w * rho))
    return out.reshape(shp)


def valid_points(atlas):
    """same valid set as ddm_ensemble_selection.select_ensemble: std, acf_1, volclust_50 non-missing (170 points)."""
    v = atlas.dropna(subset=["std", "acf_1", "volclust_50"]).copy()
    v["_key"] = list(zip(np.round(v["delta"], 6), v["wma"].astype(int)))
    return v


def cf1_selection(real, atlas, short=30, top=8):
    """CF1 (plan v0.3 3.3). Same valid set. Stage 1: 30 smallest |acf_1 - real acf_1| (ties: std_distance, delta, wma ascending). Stage 2: from the shortlist,
    8 smallest std_distance = |log std - log real std| (ties: acf_1_distance, delta, wma ascending). volclust_50 and z-scores are not used."""
    v = valid_points(atlas)
    v["acf1_dist"] = (v["acf_1"] - real["acf_1"]).abs()
    v["std_distance"] = (np.log(v["std"]) - np.log(real["std"])).abs()
    sl = v.sort_values(["acf1_dist", "std_distance", "delta", "wma"], kind="mergesort").head(short)
    sel = sl.sort_values(["std_distance", "acf1_dist", "delta", "wma"], kind="mergesort").head(top)
    return sel, sl


def key_of(delta, wma):
    return (round(float(delta), 6), int(wma))
