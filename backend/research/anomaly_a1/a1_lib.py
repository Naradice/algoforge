"""A1 (Study Manager #1140), frozen plan docs/research-anomaly-a1-plan.md v0.3 (v0.2 with the evaluation period fixed to 42 months): definitions shared by the preparation (S1) and the
evaluation (S3). The B7N forecast and the randomised PIT come from the F1 code (backend/research/f1_*.py, unchanged since db0926f)."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(RESEARCH, "..", ".."))
sys.path.insert(0, RESEARCH)
sys.path.insert(0, os.path.dirname(RESEARCH))
import f1_common as fc  # noqa: E402
import f1_models as fm  # noqa: E402

MAIN = os.path.join(REPO, "artifacts", "datasets", "derived", "usdjpy_1min_volatility.parquet")
MAIN_SHA = "c25eb5720a5552b0230fb6ce2354de1cee020e6b7ca95da46034b75f2a53c828"
F1_DIR = os.path.join(REPO, "artifacts", "research", "f1_run1_db0926f")        # durable copy of the stored F1 run
WORK = os.path.join(REPO, "artifacts", "research", "anomaly_a1")               # data arrays stay outside the repository
RESULTS = os.path.join(HERE, "results")

# ---- frozen by the plan ------------------------------------------------------------------------------------------------
fc.PERIODS["forward"] = ("2026-12-01", "2030-05-31")
FIT_START = "2017-09-01"                       # estimation without its first 12 months
LABEL_THRESHOLD = 3.50644102796436e-06         # V(t, 60) > threshold (95% point of the estimation period, fixed in A0)
MERGE_ROWS = 240
BLOCK = 60
ALPHA, ALPHA_SENS = 0.05, (0.02, 0.10)
DELTA_A = 0.05
N_MIN = 100
BOOT, BOOT_SEED = 10_000, 20261301
LEAK_N = 150
BASELINES = ("rv1", "rv4", "rv24", "rv120", "atr", "absret", "rv24_tod")      # order = tie break


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def f1_edges():
    return json.load(open(os.path.join(F1_DIR, "ddm_mapping.json")))["edges"]


def load_main():
    assert sha256(MAIN) == MAIN_SHA, "main series changed"
    df = pd.read_parquet(MAIN, columns=["close"])
    return df["close"].values.astype(float), pd.DatetimeIndex(df.index)


# ---- forecasts and PIT (same procedure as f1_run.score_month, model B7N, h = 1) --------------------------------------
def month_origins(ctx, period, month_start):
    m1 = month_start + pd.offsets.MonthBegin(1)
    in_month = np.asarray((ctx.tt >= month_start) & (ctx.tt < m1)) & ctx.origin_ok & ctx.s.period_mask(period)
    origins = np.flatnonzero(in_month)
    valid = {h: ctx.s.period_mask(period, h) & np.isfinite(ctx.targets[h]) for h in fc.HORIZONS}
    return np.array([j for j in origins if any(valid[h][j] for h in fc.HORIZONS) and valid[1][j]], dtype=int)


def pit_rng(month_start):
    return np.random.default_rng([99, fm.MODEL_IDS["B7N"], 1, month_start.month, month_start.year])


def month_pit(ctx, period, month_start, state=None):
    """(origin blocks, randomised PIT of the one-window-ahead B7N forecast) for one month."""
    st = state if state is not None else fm.build_state(ctx, month_start, None)
    js = month_origins(ctx, period, month_start)
    if len(js) == 0:
        return js, np.zeros(0)
    samples = np.array([fm.SAMPLERS["B7N"](st, ctx, j)[0] for j in js])
    assert np.all(np.isfinite(samples)), "non-finite forecast sample"
    sc = fc.score_samples(samples, ctx.targets[1][js], pit_rng(month_start))
    return js, sc["pit"]


# ---- features, labels, origin sets ------------------------------------------------------------------------------------
class Features:
    """Everything except the PIT. All score inputs at block j use rows <= t_j; the time-of-day factors use blocks that end before the origin's month."""

    def __init__(self, ctx):
        s = ctx.s
        self.ctx = ctx
        n = s.nblocks
        r = np.diff(s.logc)
        sq = np.concatenate([[0.0], np.cumsum(r * r)])                     # sq[k] = sum of squared returns up to row k
        self.sq = sq
        contig = s.contiguous(s.t - (BLOCK - 1), s.t)
        self.valid_blocks = np.flatnonzero(contig)
        rv59 = sq[s.t[self.valid_blocks]] - sq[s.t[self.valid_blocks] - (BLOCK - 1)]
        self.rv59 = rv59
        pos = np.full(n, -1)
        pos[self.valid_blocks] = np.arange(len(self.valid_blocks))
        self.pos = pos
        local = s.t_time[self.valid_blocks] - pd.Timedelta(hours=7)          # New York local time of the block end
        self.local_hour = np.asarray(local.hour)
        self.end_time = s.t_time[self.valid_blocks]
        self.sunday = np.concatenate([[0], np.cumsum(s.index.dayofweek == 6)])
        self._tod = {}

    def tod_factor(self, month_start):
        if month_start not in self._tod:
            k = int(np.sum(self.end_time < month_start))
            fac = np.ones(24)
            if k:
                overall = self.rv59[:k].mean()
                h = self.local_hour[:k]
                for x in range(24):
                    sel = h == x
                    if sel.any():
                        fac[x] = self.rv59[:k][sel].mean() / overall
            self._tod[month_start] = fac
        return self._tod[month_start]

    def scores(self, j):
        """baseline candidates (BASELINES order) and the control score at origin block j; NaN where undefined."""
        ctx = self.ctx
        out = np.full(len(BASELINES), np.nan)
        p = self.pos[j]
        control = np.nan
        if p >= 0:
            out[0] = self.rv59[p]
            for i, k in ((1, 4), (2, 24), (3, 120)):
                if p + 1 >= k:
                    out[i] = self.rv59[p + 1 - k:p + 1].mean()
            out[4] = ctx.atr[j]
            out[5] = abs(ctx.r[j])
            t = ctx.tt[j]
            fac = self.tod_factor(pd.Timestamp(t.year, t.month, 1))
            out[6] = out[2] * fac[(t + pd.Timedelta(hours=1) - pd.Timedelta(hours=7)).hour]
            q = int(np.searchsorted(ctx.valid_r, j, side="left"))             # windows with a return strictly before j
            if q >= 24 and np.isfinite(ctx.r[j]):
                sd = float(np.std(ctx.r[ctx.valid_r[q - 24:q]]))
                if sd > 0:
                    control = abs(ctx.r[j]) / sd
        return out, control

    def label_value(self, j):
        t = self.ctx.s.t[j]
        return self.sq[t + BLOCK] - self.sq[t]


def valid_origins(ctx, period):
    return ctx.origin_ok & ctx.s.period_mask(period) & ctx.s.period_mask(period, 1) & np.isfinite(ctx.targets[1])


def included_origins(F, period, drop_sunday=True, since=None):
    """plan section 2: valid origins whose previous origin is valid and (main analysis) with no Sunday-dated row in the previous window,
    the origin block or the label window (rows t-119 .. t+60). since: only origins at or after this time (the combination fit)."""
    ctx = F.ctx
    v = valid_origins(ctx, period)
    prev = np.zeros(len(v), bool)
    prev[1:] = v[:-1]
    js = np.flatnonzero(v & prev)
    if drop_sunday:
        t = ctx.s.t[js]
        lo = np.maximum(t - 119, 0)
        js = js[(F.sunday[t + BLOCK + 1] - F.sunday[lo]) == 0]
    if since is not None:
        js = js[np.asarray(ctx.tt[js] >= pd.Timestamp(since))]
    return js


def table(F, period, pit, drop_sunday=True, since=None):
    """one row per included origin: time, row, label value, positive flag, S, baselines, control. pit: dict block -> PIT."""
    ctx = F.ctx
    js = included_origins(F, period, drop_sunday, since)
    S = np.array([abs(2 * pit[j - 1] - 1) for j in js])
    pairs = [F.scores(j) for j in js]
    B = np.array([p[0] for p in pairs]).reshape(len(js), len(BASELINES))
    C = np.array([p[1] for p in pairs])
    V = np.array([F.label_value(j) for j in js])
    assert np.isfinite(S).all() and np.isfinite(B).all() and np.isfinite(C).all(), "undefined score at an included origin"
    return {"j": js, "row": ctx.s.t[js], "time": ctx.tt[js], "V": V, "y": V > LABEL_THRESHOLD, "S": S, "B": B, "C": C}


# ---- events, alerts, combination, judgement ---------------------------------------------------------------------------
def event_onsets(row, y):
    """indices (into the origin table) of the first positive origin of every event."""
    idx = np.flatnonzero(y)
    if len(idx) == 0:
        return idx
    new = np.concatenate([[True], np.diff(row[idx]) > MERGE_ROWS])
    return idx[new]


def alert_order(score):
    """positions sorted by score, largest first; ties keep time order."""
    return np.argsort(-np.asarray(score), kind="stable")


def detection_rate(order, onset_mask, alpha, weight=None):
    """share of event onsets among the alerts; with weights (bootstrap copies) the first floor(alpha * N*) copies are alerted."""
    w = np.ones(len(order)) if weight is None else weight
    ws = w[order]
    k = int(np.floor(alpha * ws.sum() + 1e-9))
    before = np.cumsum(ws) - ws
    taken = np.clip(k - before, 0, ws)
    total = w[onset_mask].sum()
    return float(taken[onset_mask[order]].sum() / total) if total > 0 else np.nan


def ecdf(reference, x):
    return np.searchsorted(reference, x, side="right") / len(reference)


def fit_logit(y, a, b):
    import statsmodels.api as sm
    X = sm.add_constant(np.column_stack([a, b]), has_constant="add")
    res = sm.Logit(y.astype(float), X).fit(disp=0, method="newton", maxiter=200)
    assert res.mle_retvals["converged"], "logistic regression did not converge"
    return np.asarray(res.params, float)


def combined_score(coef, ref_a, ref_b, a, b):
    """linear predictor of the frozen logistic regression on the estimation-period ECDF values."""
    return coef[0] + coef[1] * ecdf(ref_a, a) + coef[2] * ecdf(ref_b, b)


def week_ids(time):
    iso = (pd.DatetimeIndex(time) + pd.Timedelta(hours=6)).isocalendar()
    return iso["year"].values.astype(np.int64) * 100 + iso["week"].values.astype(np.int64)


def bootstrap(orders, onset_mask, weeks, alpha, n_boot, seed):
    """orders: dict name -> alert order. Returns array [n_boot, 3]: rates of 'base', 'base_s', 'base_c' per week-block resample."""
    uniq, inv = np.unique(weeks, return_inverse=True)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, 3))
    for b in range(n_boot):
        m = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        w = m[inv]
        out[b] = [detection_rate(orders[k], onset_mask, alpha, w) for k in ("base", "base_s", "base_c")]
    return out


def judge(ci_delta, ci_excess, n_events):
    if n_events < N_MIN:
        return "判定保留"
    if ci_delta[0] > DELTA_A and ci_excess[0] > 0:
        return "支持"
    if ci_delta[0] >= -DELTA_A and ci_delta[1] <= DELTA_A:
        return "上乗せなし"
    return "判定保留"


def compare(tab, frozen, alpha, n_boot=0):
    """primary comparison on one origin table with the frozen baseline and coefficients."""
    b = BASELINES.index(frozen["baseline"])
    base = tab["B"][:, b]
    ref = frozen["_ref"]
    scores = {"base": base,
              "base_s": combined_score(frozen["coef_s"], ref["base"], ref["S"], base, tab["S"]),
              "base_c": combined_score(frozen["coef_c"], ref["base"], ref["C"], base, tab["C"])}
    orders = {k: alert_order(v) for k, v in scores.items()}
    onsets = event_onsets(tab["row"], tab["y"])
    mask = np.zeros(len(base), bool)
    mask[onsets] = True
    rates = {k: detection_rate(orders[k], mask, alpha) for k in orders}
    out = {"alpha": alpha, "origins": int(len(base)), "positive_origins": int(tab["y"].sum()), "events": int(len(onsets)),
           "alerts": int(np.floor(alpha * len(base) + 1e-9)), "rates": rates, "delta": rates["base_s"] - rates["base"], "delta_control": rates["base_c"] - rates["base"]}
    out["excess"] = out["delta"] - out["delta_control"]
    if n_boot:
        bt = bootstrap(orders, mask, week_ids(tab["time"]), alpha, n_boot, BOOT_SEED)
        d, dc = bt[:, 1] - bt[:, 0], bt[:, 2] - bt[:, 0]
        q = lambda x: [float(v) for v in np.nanpercentile(x, [2.5, 97.5])]
        out.update({"ci_delta": q(d), "ci_delta_control": q(dc), "ci_excess": q(d - dc), "n_boot": n_boot})
        out["judgement"] = judge(out["ci_delta"], out["ci_excess"], out["events"])
    return out


# ---- leak test --------------------------------------------------------------------------------------------------------
def perturbed(close, row, rng):
    out = close.copy()
    steps = rng.normal(0, 0.002, size=len(close) - row - 1)
    out[row + 1:] = close[row] * np.exp(np.cumsum(steps))
    return out


def leak_one(close, index, edges, period, F, state_prev, j, rng):
    """True when every score at origin j (S from the forecast of origin j-1, the baselines, the control) is bit-identical after replacing all closes
    after row t_j by random values. state_prev: state of the month of origin j-1, built from the unperturbed series (state invariance is checked
    separately)."""
    ctx = F.ctx
    p = perturbed(close, int(ctx.s.t[j]), rng)
    ctx_p = fm.Context(p, index, edges)
    Fp = Features(ctx_p)
    ms = pd.Timestamp(ctx.tt[j - 1].year, ctx.tt[j - 1].month, 1)

    def pit_of(c):
        js = month_origins(c, period, ms)
        k = int(np.flatnonzero(js == j - 1)[0])
        x = fm.SAMPLERS["B7N"](state_prev, c, j - 1)[0]
        u = pit_rng(ms).random(len(js))[k]
        y = c.targets[1][j - 1]
        return (np.sum(x < y) + u * np.sum(x == y)) / len(x)

    a, ac = F.scores(j)
    b, bc = Fp.scores(j)
    return bool(pit_of(ctx) == pit_of(ctx_p) and np.array_equal(a, b) and ac == bc)


STATE_KEYS = ("har_beta", "har_resid", "z_std")


def state_unchanged(close, index, edges, month_start, state, rng):
    """the learned state of a month used by B7N does not depend on any row from the month start on."""
    first = int(np.searchsorted(index.values, np.datetime64(month_start)))
    if first >= len(close):
        return True
    p = perturbed(close, first - 1, rng)
    other = fm.build_state(fm.Context(p, index, edges), month_start, None)
    return bool(all(np.array_equal(getattr(state, k), getattr(other, k)) for k in STATE_KEYS))
