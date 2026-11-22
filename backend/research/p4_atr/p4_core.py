"""P4 shared objects (plan v0.2 @ 9d9825d): DDM per-origin evaluation (seed-wise), empirical baselines E1a/E1b/E1c, conditioning cells."""
from __future__ import annotations

import numpy as np

import p4_lib as P

SEEDS, HZ, PERIODS = P.SEEDS, P.HZ, P.PERIODS


class DdmEval:
    """Per-origin diagnostics of a stored DDM variant (3 seeds). Linear metrics (hit90, CRPS) are averaged over seeds per origin;
    nonlinear ones (k90, ...) are computed per seed and averaged (as in P3)."""

    def __init__(self, data, cond):
        self.data, self.cond = data, cond
        self.o = {}
        for s in SEEDS:
            ens = P.V.load_ens(cond, s)
            for per in PERIODS:
                for h in HZ:
                    c = data.cells[(per, h)]
                    self.o.setdefault((per, h), []).append(P.Pred.from_ens(ens[:, c["hi"], :]).per_origin(c["b"], c["y"]))
        self.crps = {k: np.mean([o["crps"] for o in v], axis=0) for k, v in self.o.items()}
        self.hit90 = {k: np.mean([o["hit90"] for o in v], axis=0) for k, v in self.o.items()}

    def metrics(self, per, h, mask=None):
        c = self.data.cells[(per, h)]
        return P.avg_metrics([P.metrics(o, c["y"], mask) for o in self.o[(per, h)]])

    def group_metrics(self, per, h, mask=None):
        c = self.data.cells[(per, h)]
        gm = [P.group_metrics(o, c["y"], c["b"], mask) for o in self.o[(per, h)]]
        return {g: P.avg_metrics([x[g] for x in gm]) for g in gm[0]}


# ---- empirical baselines (same information as the DDM: the ATR bucket) -------------------------------------------------------------
def fit_E1(kind, y, b):
    """returns list of 20 sorted sample arrays. E1a: bucket empirical y; E1c: pooled y of buckets b-1..b+1 (clipped);
    E1b: location-scale (bucket median/IQR) with the pooled standardised sample."""
    if kind == "E1a":
        return [np.sort(y[b == k]) for k in range(20)]
    if kind == "E1c":
        return [np.sort(y[(b >= max(k - 1, 0)) & (b <= min(k + 1, 19))]) for k in range(20)]
    if kind == "E1b":
        loc = np.array([np.median(y[b == k]) for k in range(20)])
        scale = np.array([np.subtract(*np.quantile(y[b == k], [0.75, 0.25])) for k in range(20)])
        if not (scale > 0).all():
            raise ValueError("non-positive IQR")
        zs = np.sort((y - loc[b]) / scale[b])
        return [loc[k] + scale[k] * zs for k in range(20)]
    raise ValueError(kind)


def fit_set(data, h, mode):
    """B-1: estimation origins; B-2: estimation + validation origins (horizon h)."""
    pers = ("estimation",) if mode == "B-1" else ("estimation", "validation")
    y = np.concatenate([data.cells[(p, h)]["y"] for p in pers])
    b = np.concatenate([data.cells[(p, h)]["b"] for p in pers])
    return y, b


def fit_cond_cells(y, cell, ncell):
    """sorted samples per conditioning cell; cells with fewer than MIN_CELL fit origins are returned as None."""
    return [np.sort(y[cell == c]) if (cell == c).sum() >= P.MIN_CELL else None for c in range(ncell)]
