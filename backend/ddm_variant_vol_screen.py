"""
Phase 8b: does any DDM variant carry predictable FUTURE volatility -- and does its volatility
behave like USDJPY's?

Phase 8a found the DDM v3_shock pretrain data has no predictable future volatility (HAR log-R^2
0.017 vs USDJPY 0.68), so pretraining on it cannot teach future-volatility forecasting. Before
building any B''' condition, screen DDMv3 variants on the same future-only target and compare
their volatility dynamics with USDJPY's.

Per variant: N_RUNS independent runs x CANDLES candles (the pretrain data's own layout), each run
its own contiguous block; windows never cross runs. Split by run (first 3/4 train, rest test).

Predictability of y_t = log std(r_{t+1} .. r_{t+20}) from the 60 returns before it:
  - persistence : log std of the last 20 input returns, unfitted
  - har         : linear regression on log std over the last 5 / 20 / 60 returns
Volatility dynamics (pooled within runs):
  - acf_abs_r   : ACF of |r| at lags 1, 5, 20, 60, 240
  - acf_logvol  : ACF of log rv20 (non-overlapping 20-bar blocks) at lags 1, 3, 12 blocks
  - decay_lag   : first lag where the |r| ACF falls below 0.05
  - har_coef    : HAR coefficients (5 / 20 / 60) and R^2 fitted on the full series
USDJPY (dataset 29, last 1M rows, gap-free windows) is scored the same way (chronological 3/4
split) as the reference.

Usage: python ddm_variant_vol_screen.py   (writes ddm_variant_vol_screen.json)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from probe_representations import _artifact_store, _dataset_artifact_path
from submit_transfer_experiment import DDM_NUM_AGENT, DDM_TRADES_PER_CANDLE

N_RUNS = 12
CANDLES = 5_000
OBS, HORIZON = 60, 20
ACF_LAGS = (1, 5, 20, 60, 240)

BASE = dict(max_volatility=0.02, min_volatility=0.01, wma=5,
            dealer_sensitive_min=-3.5, dealer_sensitive_max=-1.5)
VARIANTS = {
    "v3_shock (current)": dict(exogenous_shock_probability=0.0015, exogenous_shock_size=0.3),
    "v3 no shock": dict(),
    "decayed shock tau300": dict(exogenous_shock_probability=0.002, exogenous_shock_size=0.3,
                                 exogenous_shock_decay_tau=300.0),
    "spread feedback a0.25": dict(exogenous_shock_probability=0.0015, exogenous_shock_size=0.3,
                                  spread_feedback_a=0.25, spread_feedback_window=150),
    "loss limit 1.404": dict(exogenous_shock_probability=0.0015, exogenous_shock_size=0.3,
                             loss_limit=1.404, loss_limit_c=0.02),
}


def _simulate(kwargs: dict, seed0: int = 5000) -> list[np.ndarray]:
    from data.collectors.ddm_simulator import DDMv3

    runs = []
    for k in range(N_RUNS):
        np.random.seed(seed0 + k)
        model = DDMv3(num_agent=DDM_NUM_AGENT, **BASE, **kwargs)
        prices = model.simulate(n_trades=CANDLES * DDM_TRADES_PER_CANDLE)["price"].values
        n = (len(prices) // DDM_TRADES_PER_CANDLE) * DDM_TRADES_PER_CANDLE
        close = prices[:n].reshape(-1, DDM_TRADES_PER_CANDLE)[:, -1][:CANDLES]
        runs.append(np.diff(np.log(close)))
    return runs


def _windows(r: np.ndarray, anchors: np.ndarray):
    X = np.lib.stride_tricks.sliding_window_view(r, OBS)[anchors - OBS + 1]
    F = np.lib.stride_tricks.sliding_window_view(r, HORIZON)[anchors + 1]
    y = F.std(axis=1, ddof=1)
    feats = np.c_[X[:, -5:].std(axis=1, ddof=1), X[:, -20:].std(axis=1, ddof=1), X.std(axis=1, ddof=1)]
    keep = (y > 0) & (feats > 0).all(axis=1)
    return np.log(feats[keep]), np.log(y[keep])


def _segments_windows(segments: list[np.ndarray]):
    F, Y = [], []
    for r in segments:
        if len(r) < OBS + HORIZON + 1:
            continue
        f, y = _windows(r, np.arange(OBS - 1, len(r) - HORIZON))
        F.append(f)
        Y.append(y)
    return np.concatenate(F), np.concatenate(Y)


def _r2(y, p):
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def _predictability(train: list[np.ndarray], test: list[np.ndarray]) -> dict:
    from sklearn.linear_model import LinearRegression

    Ftr, ytr = _segments_windows(train)
    Fte, yte = _segments_windows(test)
    lr = LinearRegression().fit(Ftr, ytr)
    return {"persistence_r2_log": _r2(yte, Fte[:, 1]), "har_r2_log": _r2(yte, lr.predict(Fte)),
            "har_coef": [float(c) for c in lr.coef_], "n_test_windows": int(len(yte))}


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    return float((x[:-lag] * x[lag:]).mean() / x.var())


def _dynamics(segments: list[np.ndarray]) -> dict:
    # pooled within-segment ACFs (lags never cross a segment boundary)
    def pooled(series_list, lag):
        num = den = 0.0
        mu = np.concatenate(series_list).mean()
        for s in series_list:
            if len(s) > lag:
                d = s - mu
                num += (d[:-lag] * d[lag:]).sum()
        allv = np.concatenate(series_list) - mu
        den = (allv ** 2).sum()
        n_all = len(allv)
        n_pairs = sum(max(len(s) - lag, 0) for s in series_list)
        return float((num / max(n_pairs, 1)) / (den / n_all))

    absr = [np.abs(s) for s in segments]
    acf_abs = {lag: pooled(absr, lag) for lag in ACF_LAGS}
    decay = next((lag for lag in range(1, 481) if pooled(absr, lag) < 0.05), None)
    blocks = [np.log(s[: len(s) // 20 * 20].reshape(-1, 20).std(axis=1, ddof=1) + 1e-12)
              for s in segments if len(s) >= 40]
    acf_lv = {lag: pooled(blocks, lag) for lag in (1, 3, 12)}
    return {"acf_abs_r": acf_abs, "decay_lag_abs_r_below_0.05": decay, "acf_logvol_blocks": acf_lv,
            "kurtosis_r": float(pd.Series(np.concatenate(segments)).kurt())}


def _report(label: str, pred: dict, dyn: dict) -> None:
    acf = " ".join(f"{dyn['acf_abs_r'][l]:.3f}" for l in ACF_LAGS)
    lv = " ".join(f"{v:.3f}" for v in dyn["acf_logvol_blocks"].values())
    coef = " ".join(f"{c:.2f}" for c in pred["har_coef"])
    print(f"{label:<24}{pred['persistence_r2_log']:>9.3f}{pred['har_r2_log']:>8.3f}  [{coef}]  "
          f"|r| ACF {acf}  logvol ACF {lv}  decay={dyn['decay_lag_abs_r_below_0.05']}  "
          f"kurt={dyn['kurtosis_r']:.0f}", flush=True)


async def main() -> None:
    print(f"{'source':<24}{'persist':>9}{'HAR':>8}  [HAR coef 5/20/60]  "
          f"|r| ACF lags {ACF_LAGS}  logvol ACF lags 1,3,12 blocks")
    results = {}

    df = pd.read_parquet(_artifact_store() / await _dataset_artifact_path(29), columns=["close"]).iloc[-1_000_000:]
    r = np.diff(np.log(df["close"].to_numpy(dtype=np.float64)))
    gap = df.index.to_series().diff().dt.total_seconds().ne(60).to_numpy()[1:]
    cuts = np.flatnonzero(gap)
    usd_segments = [s for s in np.split(r, cuts) if len(s) > 0]
    usd_segments = [s[1:] if i > 0 else s for i, s in enumerate(usd_segments)]  # drop cross-gap return
    split = int(len(usd_segments) * 0.75)
    pred = _predictability(usd_segments[:split], usd_segments[split:])
    dyn = _dynamics(usd_segments)
    results["USDJPY"] = {**pred, **dyn}
    _report("USDJPY (reference)", pred, dyn)

    for label, kwargs in VARIANTS.items():
        runs = _simulate(kwargs)
        split = int(N_RUNS * 0.75)
        pred = _predictability(runs[:split], runs[split:])
        dyn = _dynamics(runs)
        results[label] = {"kwargs": kwargs, **pred, **dyn}
        _report(label, pred, dyn)

    out = Path(__file__).resolve().parent / "ddm_variant_vol_screen.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
