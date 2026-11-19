"""P3 evaluation (plan v0.3 @ c65cede, sections 9B and 10): metrics per (condition, seed, period, horizon) and paired comparisons.
Reuses the P2 library (origin sets, y = log C1 - log C0, ATR buckets, week-block bootstrap, k_c). No p-values are used for decisions."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "p2_diagnostics"))
sys.path.insert(0, os.path.join(HERE, "..", "p1_verification"))
import p2_data as D  # noqa: E402
import p2_lib as L  # noqa: E402
import p3_env as E  # noqa: E402

PERIODS = ("estimation", "validation", "final")
HZ = D.HORIZONS


def fair_crps(sorted_ens, bucket, y):
    """fair CRPS per origin (M samples): mean|x-y| - sum_k (2k-M-1) x_(k) / (M (M-1)); sorted_ens [B, M]."""
    nb, M = sorted_ens.shape
    S = np.concatenate([np.zeros((nb, 1)), np.cumsum(sorted_ens, axis=1)], axis=1)
    kk = np.arange(1, M + 1)
    pair = ((2 * kk - M - 1)[None, :] * sorted_ens).sum(axis=1) / (M * (M - 1))
    out = np.empty(len(y))
    for b in np.unique(bucket):
        m = bucket == b
        yy = y[m]
        k = np.searchsorted(sorted_ens[b], yy, side="right")
        out[m] = (yy * (2 * k - M) + S[b, M] - 2 * S[b, k]) / M - pair[b]
    return out


class Data:
    def __init__(self):
        self.raw = D.load_raw()
        self.sets = D.origin_sets(self.raw)
        self.mapping = D.load_mapping()
        self.bucket = D.buckets_for(self.raw, self.mapping)
        self.cells = {}
        for per in PERIODS:
            for hi, h in enumerate(HZ):
                j = self.sets[(per, h)]
                self.cells[(per, h)] = {"j": j, "y": D.y_target(self.raw, h, j), "b": self.bucket[j], "meta": D.cell_meta(self.raw, j), "hi": hi}
        self._wb = {}

    def wb(self, per, h, B=10000):
        key = (per, h, B)
        if key not in self._wb:
            c = self.cells[(per, h)]
            self._wb[key] = L.WeekBoot(c["meta"]["week"], B, seed=[20261301, PERIODS.index(per), c["hi"]])
        return self._wb[key]


def load_ens(cond, seed_id):
    if cond == "C0":
        e = np.load(os.path.join(E.RUN_DIR, f"ddm_ens_seed{seed_id}.npy"))
    else:
        e = np.load(os.path.join(E.ENS_DIR, f"{cond}_seed{seed_id}.npz"))["ens"]
    return np.sort(e, axis=2)


def cell_metrics(data, ens, per, h, keep_crps=True):
    c = data.cells[(per, h)]
    hi = c["hi"]
    se = ens[:, hi, :]
    o = L.origin_stats(se, c["b"], c["y"])
    y = c["y"]
    crps = fair_crps(se, c["b"], y)
    m = {"n": int(len(y)), "crps": float(crps.mean())}
    m["cov50"], m["cov80"], m["cov90"], m["cov95"] = (float(o[f"hit{k}"].mean()) for k in (50, 80, 90, 95))
    for cc in (50, 90):
        kk = L.k_c(y, o["med"], o[f"lo{cc}"], o[f"hi{cc}"], cc)
        m[f"k{cc}"] = kk["k"]
    m["P_up"] = float((y > o["med"]).mean())
    e1, e2 = L.pit_moments(o["pit_lo"], o["pit_w"])
    m["pit_mean"] = float(e1.mean())
    m["pit_var"] = float(e2.mean() - e1.mean() ** 2)
    m["exceed25_minus_nom_pp"] = float(((y < o["q0.25"]).mean() - 0.25) * 100)
    m["exceed75_minus_nom_pp"] = float(((y < o["q0.75"]).mean() - 0.75) * 100)
    g = c["b"] // 4
    m["by_atr"] = {str(a): {"n": int((g == a).sum()), "cov90": float(o["hit90"][g == a].mean()),
                            "k90": L.k_c(y[g == a], o["med"][g == a], o["lo90"][g == a], o["hi90"][g == a], 90)["k"]} for a in range(5)}
    return m, (crps if keep_crps else None)


def all_metrics(data, cond, seed_id, keep_crps=True):
    ens = load_ens(cond, seed_id)
    out, crps = {}, {}
    for per in PERIODS:
        for h in HZ:
            m, cr = cell_metrics(data, ens, per, h, keep_crps)
            out[f"{per}|h{h}"] = m
            crps[f"{per}|h{h}"] = cr
    return out, crps


def crps_rel_diff(data, per, h, crps_x, crps_ref, B=10000):
    """relative change of the mean CRPS (sum x / sum ref - 1) with a week-block bootstrap 95% interval (descriptive)."""
    wb = data.wb(per, h, B)
    th, st, _ = wb.ratio(crps_x, crps_ref)
    return {"est": float(th - 1.0), "ci": [float(np.nanquantile(st, .025) - 1), float(np.nanquantile(st, .975) - 1)]}


def stat_ref_crps(data, per, h):
    """F1 selected statistical model's per-origin CRPS (stored) for the same origins."""
    sel = json.load(open(os.path.join(E.RUN_DIR, "selection.json")))["selected"][str(h)]
    sc = np.load(os.path.join(E.RUN_DIR, f"scores_{per}.npz")) if per != "estimation" else None
    if sc is None:
        return None
    j = sc[f"{sel}|h{h}|j"]
    assert np.array_equal(j, data.cells[(per, h)]["j"])
    return sc[f"{sel}|h{h}|crps"]
