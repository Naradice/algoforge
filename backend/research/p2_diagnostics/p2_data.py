"""P2 data loading: raw data, origin sets (P1-verified rule), y = log C1 - log C0, buckets, stored ensembles."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "p1_verification"))
import p1_lib as P1  # noqa: E402
import p2_env as E  # noqa: E402

HORIZONS = P1.HORIZONS


def load_raw():
    return P1.Raw(E.PARQUET)


def y_target(raw, h, j):
    """plan section 4: y = log C1 - log C0 (NOT log(C1/C0))."""
    t = raw.t[j]
    return np.log(raw.close[t + P1.BLOCK * h]) - np.log(raw.close[t])


def y_target_ratio(raw, h, j):
    t = raw.t[j]
    return np.log(raw.close[t + P1.BLOCK * h] / raw.close[t])


def origin_sets(raw, periods=("estimation", "validation", "final")):
    ok = raw.origin_ok()
    tg = {h: raw.target(h) for h in HORIZONS}
    sets = {}
    for per in periods:
        for h in HORIZONS:
            sets[(per, h)] = np.flatnonzero(ok & raw.period_ok(per, h) & np.isfinite(tg[h]))
    return sets


def load_mapping():
    return json.load(open(os.path.join(E.RUN_DIR, "ddm_mapping.json")))


def buckets_for(raw, mapping):
    edges = np.asarray(mapping["edges"])
    return np.searchsorted(edges, raw.atr(), side="right")


def load_ens():
    """{seed: sorted ensembles [20, 4, M]}"""
    return {s: np.sort(np.load(os.path.join(E.RUN_DIR, f"ddm_ens_seed{s}.npy")), axis=2) for s in (1, 2, 3)}


def cell_meta(raw, j):
    tt = raw.t_time[j]
    return {"week": P1.week_ids(tt), "hour": np.asarray(tt.hour), "month": np.asarray(tt.year * 100 + tt.month), "time": tt}
