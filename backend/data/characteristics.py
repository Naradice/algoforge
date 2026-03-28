"""
Characteristic analysis registry — adapted from stocknet/scripts/compute_validation.py.

Each registered function takes a pandas DataFrame (OHLC, lowercase columns) and returns
a serialisable dict of metrics and plot data.

Register additional analyses:

    from .characteristics import register

    @register("my_metric")
    def compute_my_metric(df: pd.DataFrame) -> dict:
        ...

Registered names appear in GET /api/v1/data/analyses and in the dataset detail UI.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import laplace, norm, probplot
from scipy.stats import skew as _skew
from scipy.stats import kurtosis as _kurtosis
from statsmodels.tsa.stattools import acf as _acf

AnalysisFn = Callable[[pd.DataFrame], dict[str, Any]]

CHARACTERISTIC_REGISTRY: dict[str, AnalysisFn] = {}

# ── Config constants ──────────────────────────────────────────────────────────
LAGS_ACF = 50
LAGS_VOL = 100
CCDF_POINTS = 200
HIST_BINS = 100
PERIODS = [1, 2, 5, 10, 20, 50, 100, 200]


def register(name: str) -> Callable[[AnalysisFn], AnalysisFn]:
    def decorator(fn: AnalysisFn) -> AnalysisFn:
        CHARACTERISTIC_REGISTRY[name] = fn
        return fn
    return decorator


# ── Internal helpers (same as compute_validation.py) ─────────────────────────

def _returns(price: pd.Series) -> np.ndarray:
    return np.log(price).diff().dropna().values


def _ccdf(arr: np.ndarray) -> tuple[list, list]:
    abs_r = np.abs(arr)
    pos = abs_r[abs_r > 0]
    if len(pos) == 0:
        return [], []
    sorted_r = np.sort(pos)
    n = len(sorted_r)
    idx = np.unique(np.linspace(0, n - 1, min(CCDF_POINTS, n)).astype(int))
    return sorted_r[idx].tolist(), (1 - idx / n).tolist()


def _hurst(arr: np.ndarray) -> float:
    max_lag = min(50, len(arr) // 4)
    if max_lag < 2:
        return float("nan")
    lags = range(2, max_lag)
    tau = [float(np.std(arr[lg:] - arr[:-lg])) for lg in lags]
    if any(t <= 0 for t in tau):
        return float("nan")
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(poly[0])


def _acf_safe(arr: np.ndarray, nlags: int) -> list:
    actual = min(nlags, len(arr) // 2 - 1)
    if actual < 1:
        return [1.0]
    return _acf(arr, nlags=actual, fft=True).tolist()


def _vol(x: Any) -> float | None:
    return float(x) if not np.isnan(x) else None


DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _seasonality(df: pd.DataFrame, vol_col: str) -> dict:
    v = df[vol_col].astype(float)
    idx = df.index
    diffs = idx.to_series().diff().dropna()
    time_unit_min = max(1, int(diffs.median().total_seconds() / 60))
    counts_per_day = int(24 * 60 / time_unit_min)
    time_index = (idx.dayofweek * counts_per_day + idx.hour * (60 // time_unit_min) + idx.minute // time_unit_min)
    df_w = pd.DataFrame({"volume": v.values, "ti": time_index})
    seasonal_w = df_w.groupby("ti")["volume"].mean().reindex(range(counts_per_day * 7))
    weekly_days = []
    for d, label in enumerate(DAY_LABELS):
        start, end = d * counts_per_day, (d + 1) * counts_per_day
        seg = seasonal_w.iloc[start:end]
        weekly_days.append({"label": label, "slots": list(range(start, end)), "volume": [_vol(x) for x in seg.values]})
    slot_minutes = [t * time_unit_min for t in range(counts_per_day)]
    weekly = {
        "days": weekly_days,
        "counts_per_day": counts_per_day,
        "time_unit_min": time_unit_min,
        "time_labels": [f"{m // 60:02d}:{m % 60:02d}" for m in slot_minutes],
        "day_boundaries": [d * counts_per_day for d in range(8)],
    }
    week_of_month = (idx.day - 1) // 7
    df_m = pd.DataFrame({"volume": v.values, "week": week_of_month, "dow": idx.dayofweek})
    monthly_weeks = []
    for w in range(5):
        sub = df_m[df_m["week"] == w].groupby("dow")["volume"].mean().reindex(range(7))
        if sub.notna().any():
            monthly_weeks.append({"label": f"Week {w + 1}", "days": list(range(7)), "volume": [_vol(x) for x in sub.values]})
    monthly = {"weeks": monthly_weeks, "day_labels": DAY_LABELS}
    years = sorted(idx.year.unique().tolist())
    df_y = pd.DataFrame({"volume": v.values, "month": idx.month, "year": idx.year})
    yearly_series = []
    for yr in years:
        sub = df_y[df_y["year"] == yr].groupby("month")["volume"].mean().reindex(range(1, 13))
        yearly_series.append({"label": str(yr), "months": list(range(1, 13)), "volume": [_vol(x) for x in sub.values]})
    if len(years) > 1:
        overall = df_y.groupby("month")["volume"].mean().reindex(range(1, 13))
        yearly_series.insert(0, {"label": "avg", "months": list(range(1, 13)), "volume": [_vol(x) for x in overall.values]})
    return {"weekly": weekly, "monthly": monthly, "yearly": {"series": yearly_series, "month_labels": MONTH_LABELS}}


def _compute_exogenous(df: pd.DataFrame) -> dict | None:
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    except Exception:
        return None
    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", df.columns[-1])
    price = df[close_col].dropna()
    if len(price) < 10:
        return None
    returns = np.log(price).diff().dropna()
    r = returns.values
    hourly_mean = returns.groupby(returns.index.hour).mean()
    hourly_std = returns.groupby(returns.index.hour).std().fillna(0)
    std = float(np.std(r))
    sorted_r = np.sort(r)
    n = len(sorted_r)
    idx = np.unique(np.linspace(0, n - 1, min(300, n)).astype(int))
    window = max(10, len(r) // 20)
    rolling_vals = pd.Series(r).rolling(window).mean().dropna().values
    step = max(1, len(rolling_vals) // 400)
    sampled = rolling_vals[::step]
    nlags = min(200, len(r) // 2 - 1)
    acf_vals = _acf_safe(r, nlags)
    vol_col = col_map.get("tick_volume") or col_map.get("volume")
    seasonality = None
    if vol_col and len(df.index.to_series().diff().dropna()) > 0:
        seasonality = _seasonality(df, vol_col)
    return {
        "seasonality": seasonality,
        "intraday_seasonality": {
            "hours": hourly_mean.index.tolist(),
            "mean": [float(v) for v in hourly_mean.values],
            "std": [float(v) for v in hourly_std.values],
        },
        "jump_tail": {
            "jump_rate": float(np.mean(np.abs(r) > 3 * std)),
            "threshold_3sigma": 3 * std,
            "q99": float(np.quantile(r, 0.99)),
            "q999": float(np.quantile(r, 0.999)),
            "q001": float(np.quantile(r, 0.001)),
            "q01": float(np.quantile(r, 0.01)),
        },
        "cdf": {"x": sorted_r[idx].tolist(), "y": (idx / n).tolist()},
        "rolling_mean": {"index": list(range(len(sampled))), "values": [float(v) for v in sampled], "window": window},
        "long_lag_acf": {
            "lags": list(range(len(acf_vals))),
            "values": acf_vals,
            "highlights": {str(lg): float(acf_vals[lg]) for lg in [10, 20, 30, 40, 50] if lg < len(acf_vals)},
        },
    }


# ── Registered analyses ───────────────────────────────────────────────────────

@register("full")
def compute_full(df: pd.DataFrame) -> dict:
    """
    Full stylised-facts analysis: ACF, CCDF, Hurst, fat tails, volatility clustering,
    return distribution, QQ plot, diffusion, seasonality.
    Adapted from stocknet/scripts/compute_validation.py::compute().
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", df.columns[-1])
    price = df[close_col].dropna()
    if len(price) < 20:
        raise ValueError("Need at least 20 rows for characteristic analysis")

    r = _returns(price)
    ccdf_x, ccdf_y = _ccdf(r)
    acf_r = _acf_safe(r, LAGS_ACF)
    acf_abs = _acf_safe(np.abs(r), LAGS_ACF)
    safe_periods = [int(p) for p in PERIODS if int(p) < len(price) // 2]
    vars_ = [float(np.var(price.diff(p).dropna())) for p in safe_periods]
    base = vars_[0] if vars_ and vars_[0] > 0 else 1.0
    diff_vars = [v / base for v in vars_]
    (theoretical, sample), (slope, intercept, _) = probplot(r, dist="norm")
    step = max(1, len(theoretical) // 200)
    qq = [{"t": float(t), "s": float(s)} for t, s in zip(theoretical[::step], sample[::step])]
    acf_vol = _acf_safe(np.abs(r), LAGS_VOL)
    counts, bin_edges = np.histogram(r, bins=HIST_BINS, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_r, std_r = float(r.mean()), float(r.std())
    b_laplace = std_r / np.sqrt(2)
    hist7, edges7 = np.histogram(np.abs(r), bins=HIST_BINS, density=True)
    cdf7 = np.cumsum(hist7 * np.diff(edges7))
    return {
        "stats": {
            "n": int(len(price)),
            "mean": mean_r,
            "std": std_r,
            "skewness": float(_skew(r)),
            "kurtosis": float(_kurtosis(r, fisher=False)),
            "hurst": _hurst(r),
        },
        "ccdf": {"x": ccdf_x, "y": ccdf_y},
        "acf_returns": acf_r,
        "acf_abs_returns": acf_abs,
        "diffusion": {"lags": safe_periods, "vars": diff_vars},
        "qq": qq,
        "qq_line": {"slope": float(slope), "intercept": float(intercept)},
        "volatility_clustering": acf_vol,
        "return_dist": {
            "centers": centers.tolist(),
            "hist": [float(v) if v > 0 else None for v in counts.tolist()],
            "normal_pdf": norm.pdf(centers, loc=mean_r, scale=std_r).tolist(),
            "laplace_pdf": laplace.pdf(centers, loc=mean_r, scale=b_laplace).tolist(),
        },
        "ccdf_hist": {
            "x": edges7[1:].tolist(),
            "y": [float(v) if v > 0 else None for v in (1 - cdf7).tolist()],
        },
        "exogenous": _compute_exogenous(df),
    }


@register("basic_stats")
def compute_basic_stats(df: pd.DataFrame) -> dict:
    """Row count, date range, basic OHLC stats. Fast, no heavy computation."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", df.columns[-1])
    price = df[close_col].dropna()
    r = _returns(price)
    return {
        "row_count": int(len(df)),
        "from_ts": str(df.index.min()),
        "to_ts": str(df.index.max()),
        "close_min": float(price.min()),
        "close_max": float(price.max()),
        "close_mean": float(price.mean()),
        "return_std": float(np.std(r)) if len(r) > 0 else None,
        "hurst": _hurst(r) if len(r) > 4 else None,
    }


def compute_for_dataset(artifact_path: str) -> dict:
    """
    Load a dataset parquet file and run all registered analyses.
    Returns dict keyed by analysis name.
    """
    from pathlib import Path
    import os
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    df = pd.read_parquet(store / artifact_path)
    results: dict[str, Any] = {}
    for name, fn in CHARACTERISTIC_REGISTRY.items():
        try:
            results[name] = fn(df)
        except Exception as e:
            results[name] = {"error": str(e)}
    return results
