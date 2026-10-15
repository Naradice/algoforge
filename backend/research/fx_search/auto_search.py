"""
Automated candidate search over portfolio definitions with multiple-testing control.

For every configuration in a grid we run the portfolio engine, then
  * select on the in-sample period (<= IS_END) and evaluate the *same* config out-of-sample,
  * compute the Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) of the best IS config using the number of
    trials actually run, skew and kurtosis of its returns,
  * estimate the probability of backtest overfitting as the share of CSCV-style splits in which the IS-best
    config falls below the OOS median (simple 2-fold version over calendar-year blocks).

Usage: python auto_search.py [--quick]   -> artifacts/research/fx/auto_search_results.parquet + printed summary
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
import portfolio_check as pc  # builds the 9-major universe definition (merged MT5 + yfinance D1)
from strategy.portfolio.engine import run_portfolio_backtest

IS_END = "2015-12-31"
OUT = HERE.parents[3] / "artifacts" / "research" / "fx" / "auto_search_results.parquet"


def sharpe_daily(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 50 and r.std() > 0 else float("nan")


def deflated_sharpe(sr_daily: float, n_obs: int, skew: float, kurt: float, n_trials: int, sr_var: float) -> float:
    """P(true SR > 0 | best of n_trials). sr_daily: per-period (non-annualised) Sharpe of the selected config,
    sr_var: variance of the per-period Sharpe across all trials."""
    if n_trials < 2 or n_obs < 30 or not np.isfinite(sr_daily):
        return float("nan")
    emc = 0.5772156649
    sr0 = math.sqrt(max(sr_var, 1e-12)) * ((1 - emc) * stats.norm.ppf(1 - 1.0 / n_trials)
                                           + emc * stats.norm.ppf(1 - 1.0 / (n_trials * math.e)))
    denom = math.sqrt(max(1 - skew * sr_daily + (kurt - 1) / 4.0 * sr_daily ** 2, 1e-9))
    return float(stats.norm.cdf((sr_daily - sr0) * math.sqrt(n_obs - 1) / denom))


def grid(quick: bool):
    g = {
        "threshold": [0.005, 0.01, 0.02, 0.03],
        "scale": [0.02, 0.03, 0.05],
        "cap": [1.0, 1.5, 2.5],
        "rebalance": ["M", "W"],
        "vix_cut": [0.0, 0.5, 0.7, 1.0],
        "vix_ratio": [1.2, 1.3, 1.5],
        "vol_window": [20, 60, 120],
        "mom_weight": [0.0, 0.3],
        "markup": [0.015],
    }
    if quick:
        g = {k: v[:2] for k, v in g.items()}
        g["markup"] = [0.015]
    keys = list(g)
    for vals in itertools.product(*[g[k] for k in keys]):
        c = dict(zip(keys, vals))
        if c["vix_cut"] == 0.0 and c["vix_ratio"] != g["vix_ratio"][0]:
            continue                      # ratio is irrelevant without a cut: dedupe
        yield c


def build(c: dict) -> dict:
    d = copy.deepcopy(pc.defn)
    d["sleeves"] = [{"type": "carry", "weight": 1.0 - c["mom_weight"], "threshold": c["threshold"], "scale": c["scale"],
                     "cap": c["cap"], "rebalance": c["rebalance"]}]
    if c["mom_weight"] > 0:
        d["sleeves"].append({"type": "momentum", "weight": c["mom_weight"], "lookbacks": [60, 250]})
    d["overlays"] = ([{"type": "series_jump", "path": "research/fx/rates/VIXCLS.csv", "column": "VIXCLS",
                       "ratio": c["vix_ratio"], "lookback": 20, "cut": c["vix_cut"]}] if c["vix_cut"] > 0 else [])
    d["risk"].update({"swap_markup": c["markup"], "vol_window": c["vol_window"]})
    return d


def main(quick: bool = False) -> None:
    cfgs = list(grid(quick))
    print(f"{len(cfgs)} configurations", flush=True)
    rets, rows = {}, []
    t0 = time.time()
    for i, c in enumerate(cfgs):
        r = run_portfolio_backtest(build(c))["returns"]
        key = json.dumps(c, sort_keys=True)
        rets[key] = r
        rows.append({**c, "key": key, "sr_is": sharpe_daily(r[:IS_END]), "sr_oos": sharpe_daily(r[IS_END:]),
                     "sr_all": sharpe_daily(r), "dd": float((r.cumsum() - r.cumsum().cummax()).min())})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(cfgs)}  {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)

    best = df.sort_values("sr_is", ascending=False).iloc[0]
    r_is = rets[best["key"]][:IS_END].dropna()
    sr_is_d = r_is.mean() / r_is.std()
    all_sr_d = np.array([(rets[k][:IS_END].dropna().mean() / rets[k][:IS_END].dropna().std()) for k in df["key"]])
    dsr = deflated_sharpe(sr_is_d, len(r_is), float(stats.skew(r_is)), float(stats.kurtosis(r_is, fisher=False)),
                          len(df), float(np.nanvar(all_sr_d)))
    # PBO (2-fold over year parity blocks): IS-best on even years vs its rank on odd years and vice versa
    yrs = pd.Series(rets[df["key"].iloc[0]].index.year, index=rets[df["key"].iloc[0]].index)
    pbo_hits, trials = 0, 0
    for a, b in ((0, 1), (1, 0)):
        in_mask = (yrs % 2 == a).to_numpy(); out_mask = (yrs % 2 == b).to_numpy()
        srs_in = {k: sharpe_daily(rets[k][in_mask]) for k in df["key"]}
        srs_out = {k: sharpe_daily(rets[k][out_mask]) for k in df["key"]}
        kb = max(srs_in, key=lambda k: srs_in[k] if np.isfinite(srs_in[k]) else -9)
        rank = pd.Series(srs_out).rank(pct=True)[kb]
        pbo_hits += int(rank < 0.5); trials += 1
    print("\n== summary ==")
    print(f"IS-best SR(IS)={best['sr_is']:.2f}  SR(OOS {IS_END[:4]}+)={best['sr_oos']:.2f}  config={best['key']}")
    print(f"corr(SR_IS, SR_OOS) over configs = {df['sr_is'].corr(df['sr_oos']):.2f}")
    print(f"OOS SR: median {df['sr_oos'].median():.2f}  p10 {df['sr_oos'].quantile(.1):.2f}  p90 {df['sr_oos'].quantile(.9):.2f}")
    print(f"Deflated Sharpe (IS best of {len(df)} trials): P(SR>0) = {dsr:.2f}")
    print(f"PBO (2-fold parity blocks): {pbo_hits}/{trials} folds with IS-best below OOS median")
    top = df.sort_values("sr_is", ascending=False).head(5)[["threshold","scale","cap","rebalance","vix_cut","vix_ratio",
                                                              "vol_window","mom_weight","sr_is","sr_oos","dd"]]
    print(top.round(3).to_string(index=False))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
