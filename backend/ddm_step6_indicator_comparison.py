"""Step 6 Phase A: what does the Transformer latent regime capture, versus conventional regime
indicators -- and does any conventional indicator predict future volatility as well?

Cheap: one model load (for the Transformer delta), everything else is close-only indicator math
on the same out-of-sample windows the rest of this investigation uses.

Indicators, all on the same OBS_LEN=60-candle window as the Transformer:
  Volatility    rolling_std   std of log-returns in the window
  Volatility    atr_proxy     mean |Δclose| in the window (close-only ATR: with no high/low,
                              true range degenerates to |close_t - close_{t-1}|)
  Trend         ma_slope      OLS slope of the 60-candle SMA of log-price, regressed over the
                              window's 61 points (log units -> scale-free; smoother than a raw
                              OLS on price, and NOT MA[t]-MA[t-N])
  Trend strength adx_proxy    close-only ADX: +DM/-DM from close-to-close up/down moves, Wilder-
                              smoothed period 14, DX = |+DI - -DI|/(+DI + -DI), ADX = smoothed DX
  Dynamics      acf1          lag-1 autocorrelation of log-returns in the window

Transformer: existing predicted delta -> per-window value (same as §05 / ddm_horizon_analysis).

Reports: (3) each indicator vs Transformer delta -- Spearman + Pearson; (4) each indicator vs
FUTURE realized volatility at h=1/4/12/24 -- Spearman + p (the §05 evaluation); plus a quick
check that none predict future signed return.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from ddm_ensemble_selection import (
    CALIB_FRAC, N_ROWS, OBS_LEN, START_ROW, USDJPY_ARTIFACT, predict_all_windows,
)
from ddm_horizon_analysis import HORIZONS_WINDOWS, MAX_HORIZON
from real_data_transfer_test import load_models

CHAR_MARGIN = 200  # extra candles before the first test window for indicator warmup (SMA/ADX)


def _wilder_smooth(x: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) < period:
        return out
    out[period - 1] = np.nanmean(x[:period])
    for i in range(period, len(x)):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return out


def adx_proxy_series(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Close-only ADX. Returns an array aligned to `close` (NaN during warmup)."""
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    tr = np.abs(d)
    atr = _wilder_smooth(tr, period)
    pdi = 100 * _wilder_smooth(up, period) / atr
    ndi = 100 * _wilder_smooth(dn, period) / atr
    denom = pdi + ndi
    dx = 100 * np.abs(pdi - ndi) / np.where(denom == 0, np.nan, denom)
    return _wilder_smooth(np.nan_to_num(dx, nan=0.0), period)


def main():
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    df = pd.read_parquet(store / USDJPY_ARTIFACT)
    from ddm_ensemble_selection import CHAR_WINDOW
    closes_ext = df["close"].values[START_ROW : START_ROW + N_ROWS + CHAR_WINDOW].astype(np.float64)
    closes = closes_ext[:N_ROWS]
    log_close = np.log(closes)
    log_returns = np.diff(log_close)
    mu, sigma = log_returns.mean(), log_returns.std()

    print("Loading models for Transformer delta (single load)...")
    models = asyncio.run(load_models())
    delta_sliding = predict_all_windows(models, log_returns, mu, sigma)

    n_windows = len(log_returns) // OBS_LEN
    window_start_idx = np.arange(n_windows) * OBS_LEN
    delta_per_window = delta_sliding[window_start_idx]
    n_calib = int(n_windows * CALIB_FRAC)
    test_idx = np.arange(n_calib, n_windows - MAX_HORIZON)

    # --- series-level indicators (aligned to `closes` / candle index) ---
    sma_logp = pd.Series(log_close).rolling(OBS_LEN).mean().values
    adx_s = adx_proxy_series(closes, 14)

    def window_indicators(i: int) -> dict:
        """Indicators for the window of log_returns[i*60 : i*60+60] (closes[i*60 : i*60+61])."""
        c0, c1 = i * OBS_LEN, i * OBS_LEN + OBS_LEN
        r = log_returns[c0:c1]
        px_abs = np.abs(np.diff(closes[c0 : c1 + 1]))
        # ma_slope: OLS slope of the SMA of log-price over the window's 61 points
        seg = sma_logp[c0 : c1 + 1]
        if np.any(np.isnan(seg)):
            ma_slope = np.nan
        else:
            t = np.arange(len(seg))
            ma_slope = np.polyfit(t, seg, 1)[0]
        acf1 = np.nan
        if len(r) > 2 and r.std() > 0:
            acf1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        return {
            "rolling_std": float(r.std()),
            "atr_proxy": float(px_abs.mean()),
            "ma_slope": float(ma_slope),
            "adx_proxy": float(adx_s[c1]) if c1 < len(adx_s) else np.nan,
            "acf1": acf1,
        }

    # future realized vol at horizon h for window i = windowed vol of window (i+h)
    def windowed_vol(j: int) -> float:
        s = window_start_idx[j]
        wr = np.diff(log_close[s : s + OBS_LEN + 1])
        return float(wr.std()) if len(wr) > 1 else np.nan

    def fwd_signed_return(i: int, h: int) -> float:
        return float(log_close[window_start_idx[i + h]] - log_close[window_start_idx[i]])

    ind_names = ["rolling_std", "atr_proxy", "ma_slope", "adx_proxy", "acf1"]
    rows = {name: [] for name in ind_names}
    delta_test = []
    fwd_vol = {h: [] for h in HORIZONS_WINDOWS}
    fwd_ret = {h: [] for h in HORIZONS_WINDOWS}
    for i in test_idx:
        wi = window_indicators(i)
        for name in ind_names:
            rows[name].append(wi[name])
        delta_test.append(delta_per_window[i])
        for h in HORIZONS_WINDOWS:
            fwd_vol[h].append(windowed_vol(i + h))
            fwd_ret[h].append(fwd_signed_return(i, h))
    delta_test = np.array(delta_test)
    ind_arr = {name: np.array(v) for name, v in rows.items()}
    fwd_vol = {h: np.array(v) for h, v in fwd_vol.items()}
    fwd_ret = {h: np.array(v) for h, v in fwd_ret.items()}

    RESULTS = {"n_test_windows": int(len(test_idx))}
    print(f"\nn_test_windows = {len(test_idx)}")

    # --- (3) each indicator vs Transformer delta ---
    print("\n=== (3) Indicator vs Transformer delta ===")
    print(f"{'indicator':>14} {'Spearman':>10} {'Pearson':>10}")
    RESULTS["vs_transformer_delta"] = {}
    for name in ind_names:
        m = np.isfinite(ind_arr[name]) & np.isfinite(delta_test)
        rs, _ = spearmanr(ind_arr[name][m], delta_test[m])
        rp, _ = pearsonr(ind_arr[name][m], delta_test[m])
        print(f"{name:>14} {rs:>10.3f} {rp:>10.3f}")
        RESULTS["vs_transformer_delta"][name] = {"spearman": float(rs), "pearson": float(rp), "n": int(m.sum())}

    # --- (4) each indicator (and Transformer delta) vs FUTURE realized vol ---
    print("\n=== (4) Indicator vs FUTURE realized volatility (Spearman, p) ===")
    print(f"{'indicator':>14} " + " ".join(f"{'h='+str(h):>16}" for h in HORIZONS_WINDOWS))
    RESULTS["vs_future_vol"] = {}
    series = {"transformer_delta": delta_test, **ind_arr}
    for name, arr in series.items():
        cells, rec = [], {}
        for h in HORIZONS_WINDOWS:
            m = np.isfinite(arr) & np.isfinite(fwd_vol[h])
            rs, p = spearmanr(arr[m], fwd_vol[h][m])
            cells.append(f"{rs:+.3f} (p={p:.1e})")
            rec[str(h)] = {"spearman": float(rs), "p": float(p)}
        print(f"{name:>14} " + " ".join(f"{c:>16}" for c in cells))
        RESULTS["vs_future_vol"][name] = rec

    # --- quick: none predict future signed return ---
    print("\n=== (aux) Indicator vs FUTURE signed return (Spearman, p) -- expect ~none ===")
    RESULTS["vs_future_return"] = {}
    for name, arr in series.items():
        cells, rec = [], {}
        for h in HORIZONS_WINDOWS:
            m = np.isfinite(arr) & np.isfinite(fwd_ret[h])
            rs, p = spearmanr(arr[m], fwd_ret[h][m])
            cells.append(f"{rs:+.3f} (p={p:.2f})")
            rec[str(h)] = {"spearman": float(rs), "p": float(p)}
        print(f"{name:>14} " + " ".join(f"{c:>16}" for c in cells))
        RESULTS["vs_future_return"][name] = rec

    out = Path(__file__).resolve().parent / "results" / "ddm_step6_indicator_comparison.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
