"""F1 (#889 v0.3): DDM mapping rebuilt from ESTIMATION-period windows only, and the DDM ensembles.

Frozen (unchanged from the earlier pipeline): atlas ddm_regime_atlas.csv, select_ensemble (std-first shortlist, TOP_K=8),
20 quantiles, §11 decayed shock (P0=0.002, S=0.3, tau=300, floor=0), 20 trades = 1 candle, simulate_long_path (N=300).
Rebuilt: bucket edges and bucket -> candidate-set mapping use estimation-period origins whose forward 1 001 closes
(gap-free, ending inside the estimation period) are the only 'future' statistics used.

Stages:  python research/f1_ddm.py map  <workdir>
         python research/f1_ddm.py ens  <workdir> <seed_id>      (seed_id 1..3; 20 buckets x 1000 paths)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import f1_common as fc  # noqa: E402

TRADES_PER_CANDLE = 20
CHAR_WINDOW = 1000
MAX_PER_BUCKET = 300
MAP_SEED = 20261117
N_AGENT = 300
SHOCK = dict(exogenous_shock_probability=0.002, exogenous_shock_size=0.3,
             exogenous_shock_decay_tau=300.0, exogenous_shock_floor_probability=0.0)
N_PATHS = 1000
MAX_ATTEMPTS = 4000


def simulate_long_path(delta: float, wma: int, seed: int):
    """Replica of ddm_horizon_analysis.simulate_long_path with the §11 shock (24 windows of candles, N=300)."""
    from data.collectors.ddm_simulator import DDMv3
    n_trades = fc.MAX_H * fc.BLOCK * TRADES_PER_CANDLE
    random.seed(seed)
    np.random.seed(seed)
    model = DDMv3(num_agent=N_AGENT, max_volatility=0.02, min_volatility=0.01, wma=wma,
                  dealer_sensitive_min=-3.5 + delta, dealer_sensitive_max=-1.5 + delta,
                  spread_feedback_a=None, spread_feedback_window=150, **SHOCK)
    try:
        prices = model.simulate(n_trades=n_trades)["price"].values
    except (ValueError, RuntimeError):
        return None
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        return None
    n = (len(prices) // TRADES_PER_CANDLE) * TRADES_PER_CANDLE
    closes = prices[:n].reshape(-1, TRADES_PER_CANDLE)[:, -1]
    if len(closes) < fc.MAX_H * fc.BLOCK:
        return None
    return closes[: fc.MAX_H * fc.BLOCK]


def cum_returns(closes):
    return [float(np.log(closes[fc.BLOCK * h - 1] / closes[0])) for h in fc.HORIZONS]


def bucket_job(args):
    q, cands, seed_id = args
    out, attempts = [], 0
    while len(out) < N_PATHS and attempts < MAX_ATTEMPTS:
        delta, wma = cands[attempts % len(cands)]
        seed = 8_000_000 + seed_id * 1_000_000 + q * 20_000 + attempts
        attempts += 1
        closes = simulate_long_path(delta, wma, seed)
        if closes is not None:
            out.append(cum_returns(closes))
    return q, np.array(out), attempts


def stage_map(workdir):
    os.makedirs(workdir, exist_ok=True)
    from ddm_regime_atlas import full_characterize
    from ddm_ensemble_selection import select_ensemble, TOP_K
    atlas = pd.read_csv(os.path.join(os.path.dirname(HERE), "ddm_regime_atlas.csv"))
    s = fc.load_series()
    atr = s.atr_window()
    est = s.period_mask("estimation") & s.atr_valid() & np.isfinite(atr)
    j_all = np.flatnonzero(est)
    edges = np.quantile(atr[j_all], np.linspace(0, 1, fc.N_QUANTILES + 1)[1:-1])
    bucket = np.digitize(atr[j_all], edges)
    # forward window rows [60j, 60j+1000] must be gap-free and end inside the estimation period
    a = j_all * fc.BLOCK
    b = a + CHAR_WINDOW
    ok = s.contiguous(a, b)
    est_hi = pd.Timestamp(fc.PERIODS["estimation"][1]) + pd.Timedelta(days=1)
    ok &= np.asarray(s.index[np.minimum(b, len(s.close) - 1)] < est_hi)
    rng = np.random.default_rng(MAP_SEED)
    results = []
    t0 = time.time()
    for q in range(fc.N_QUANTILES):
        idx = np.flatnonzero((bucket == q) & ok)
        n_avail = len(idx)
        if n_avail > MAX_PER_BUCKET:
            idx = rng.choice(idx, size=MAX_PER_BUCKET, replace=False)
        stats = []
        for i in idx:
            wc = s.close[a[i]:a[i] + CHAR_WINDOW + 1]
            stats.append(full_characterize(pd.DataFrame({"close": wc})))
        names = ["std", "skewness", "kurtosis", "acf_1", "acf_5", "acf_20", "volclust_50"]
        avg = {nm: float(np.nanmean([x[nm] for x in stats])) for nm in names}
        sel = select_ensemble(pd.Series(avg), atlas)
        cands = [(float(r["delta"]), int(r["wma"])) for _, r in sel["candidates"].iterrows()]
        results.append({"q": q, "n_windows_available": int(n_avail), "n_windows_used": int(len(idx)), "real_stats": avg,
                        "candidates": cands, "best_std_ratio": float(sel["best_std_ratio"]),
                        "out_of_range": bool(sel["out_of_range"])})
        print(f"q={q:2d} n={len(idx):3d}/{n_avail:4d} real_std={avg['std']:.6f} best={cands[0]} ratio={sel['best_std_ratio']:.2f} "
              f"OOR={sel['out_of_range']}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"edges": edges.tolist(), "buckets": results, "top_k": TOP_K}, open(os.path.join(workdir, "ddm_mapping.json"), "w"), indent=1)


def stage_ens(workdir, seed_id):
    mp = json.load(open(os.path.join(workdir, "ddm_mapping.json")))
    jobs = [(b["q"], [tuple(c) for c in b["candidates"]], seed_id) for b in mp["buckets"]]
    ens = np.full((fc.N_QUANTILES, len(fc.HORIZONS), N_PATHS), np.nan)
    info = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=18) as ex:
        for q, arr, att in ex.map(bucket_job, jobs):
            if len(arr) >= N_PATHS:
                ens[q] = arr[:N_PATHS].T
            else:
                ens[q, :, :len(arr)] = arr.T
            info[q] = {"successes": int(len(arr)), "attempts": int(att)}
            print(f"seed {seed_id} bucket {q}: {len(arr)} paths / {att} attempts ({time.time() - t0:.0f}s)", flush=True)
    np.save(os.path.join(workdir, f"ddm_ens_seed{seed_id}.npy"), ens)
    json.dump(info, open(os.path.join(workdir, f"ddm_ens_seed{seed_id}.json"), "w"))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "map":
        stage_map(sys.argv[2])
    else:
        stage_ens(sys.argv[2], int(sys.argv[3]))
