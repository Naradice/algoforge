"""P3 path generation (plan v0.3 @ c65cede, sections 4-5).

DDMv3S subclasses DDMv3 and overrides only _maybe_exogenous_shock (simulator source untouched):
  legacy mode    (shock_rng=None): identical to DDMv3 (global `random`).
  separated mode (shock_rng=random.Random(seed)): two draws from shock_rng at EVERY step (also when the probability is 0 or no
  shock fires), so that the global `random` stream (time noise) is identical across shock treatments.
Window: the first 20*(B+1) trades are burn-in; its last close is the window base C_B; the next 1440 candles follow;
y_h^(B) = log C[B+60h] - log C[B], same as F1 for B = 0 (closes[60h] / closes[0])."""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from data.collectors.ddm_simulator import DDMv3  # noqa: E402

TPC = 20                      # trades per candle
BLOCK = 60
HORIZONS = (1, 4, 12, 24)
MAX_H = 24
N_AGENT = 300
SHOCK = dict(exogenous_shock_probability=0.002, exogenous_shock_size=0.3, exogenous_shock_decay_tau=300.0,
             exogenous_shock_floor_probability=0.0)
N_PATHS = 1000
MAX_ATTEMPTS = 4000
SHOCK_SEED_OFFSET = 500_000_000
MAPPING = r"C:\tmp\f1_run1_db0926f\ddm_mapping.json"


class DDMv3S(DDMv3):
    def __init__(self, *args, shock_rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._shock_rng = shock_rng

    def _maybe_exogenous_shock(self):
        if self._shock_rng is None:
            return super()._maybe_exogenous_shock()
        u1 = self._shock_rng.random()
        u2 = self._shock_rng.random()
        p = self._current_shock_probability() if self.exogenous_shock_probability > 0.0 else 0.0
        if u1 >= p:
            return None
        return (1 if u2 < 0.5 else -1), self.exogenous_shock_size


# condition name -> (B, burn-in shock on, prediction-period shock on, reset clock)
CONDS = {
    "C0s": (0, True, True, False),
    "C3": (0, False, False, False),
    "C1_15": (15, True, True, False), "C1_60": (60, True, True, False), "C1_240": (240, True, True, False),
    "C2_15": (15, True, True, True), "C2_60": (60, True, True, True), "C2_240": (240, True, True, True),
    "C4_15": (15, False, False, False), "C4_60": (60, False, False, False), "C4_240": (240, False, False, False),
    "C5_15": (15, True, False, False), "C5_60": (60, True, False, False), "C5_240": (240, True, False, False),
    "C0": (0, True, True, False),          # F1 legacy mode (reproduction gate only)
}


def build_model(delta, wma, seed, burn_shock, sep, shock_override=None):
    random.seed(seed)
    np.random.seed(seed)
    kw = dict(SHOCK)
    if shock_override:
        kw.update(shock_override)
    if not burn_shock:
        kw["exogenous_shock_probability"] = 0.0
    srng = random.Random(seed + SHOCK_SEED_OFFSET) if sep else None
    return DDMv3S(num_agent=N_AGENT, max_volatility=0.02, min_volatility=0.01, wma=wma,
                  dealer_sensitive_min=-3.5 + delta, dealer_sensitive_max=-1.5 + delta,
                  spread_feedback_a=None, spread_feedback_window=150, shock_rng=srng, **kw)


def _closes(prices):
    n = (len(prices) // TPC) * TPC
    return prices[:n].reshape(-1, TPC)[:, -1]


def gen_closes(delta, wma, seed, cond, sep=True, shock_override=None):
    """returns (window closes [1441], info) or (None, None). info = dict(closes_full, model)."""
    B, burn, pred, reset = CONDS[cond]
    model = build_model(delta, wma, seed, burn, sep, shock_override)
    try:
        if burn == pred and not reset:
            n_candles = B + MAX_H * BLOCK + 1
            prices = model.simulate(n_trades=n_candles * TPC)["price"].values
            if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
                return None, None
            c = _closes(prices)
            if len(c) < n_candles:
                return None, None
            return c[B:B + MAX_H * BLOCK + 1], {"closes_full": c[:n_candles], "model": model}
        p1 = model.simulate(n_trades=TPC * (B + 1))["price"].values
        if reset:
            model._step_count = 0
        if burn and not pred:
            model.exogenous_shock_probability = 0.0
        p2 = model.simulate(n_trades=TPC * MAX_H * BLOCK)["price"].values
        prices = np.concatenate([p1, p2])
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            return None, None
        c2 = _closes(p2)
        if len(c2) < MAX_H * BLOCK:
            return None, None
        win = np.concatenate([[p1[-1]], c2[:MAX_H * BLOCK]])
        return win, {"closes_full": np.concatenate([_closes(p1), c2]), "model": model}
    except (ValueError, RuntimeError):
        return None, None


def cum_returns(win):
    return [float(np.log(win[BLOCK * h] / win[0])) for h in HORIZONS]


def bucket_job(args):
    q, cands, seed_id, cond, sep, n_paths = args
    out, ids, attempts = [], [], 0
    while len(out) < n_paths and attempts < MAX_ATTEMPTS:
        delta, wma = cands[attempts % len(cands)]
        seed = 8_000_000 + seed_id * 1_000_000 + q * 20_000 + attempts
        win, _ = gen_closes(delta, wma, seed, cond, sep=sep)
        if win is not None:
            out.append(cum_returns(win))
            ids.append(attempts)
        attempts += 1
    return q, np.array(out), np.array(ids), attempts


def make_ensemble(cond, seed_id, workers=18, buckets=None, n_paths=N_PATHS, sep=True, mapping_path=MAPPING):
    from concurrent.futures import ProcessPoolExecutor
    mp = json.load(open(mapping_path))
    qs = list(range(20)) if buckets is None else list(buckets)
    jobs = [(b["q"], [tuple(c) for c in b["candidates"]], seed_id, cond, sep, n_paths) for b in mp["buckets"] if b["q"] in qs]
    ens = np.full((20, len(HORIZONS), n_paths), np.nan)
    att_ids = np.full((20, n_paths), -1, dtype=np.int64)
    info = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for q, arr, ids, att in ex.map(bucket_job, jobs):
            k = min(len(arr), n_paths)
            ens[q, :, :k] = arr[:k].T
            att_ids[q, :k] = ids[:k]
            info[q] = {"successes": int(len(arr)), "attempts": int(att)}
            print(f"{cond} seed {seed_id} bucket {q}: {len(arr)} paths / {att} attempts", flush=True)
    return ens, att_ids, info
