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
Each registered function is run independently so results stream to the UI one at a time.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

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


# ── Internal helpers ──────────────────────────────────────────────────────────

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


def _seasonality(df: pd.DataFrame, vol_col: str | None, ret_aligned: pd.Series) -> dict:
    """
    Compute seasonality at four time scales: intraday×weekday (weekly), weekday×week-of-month
    (monthly), day-of-month×year (day_of_month), and month×year (yearly).
    Each series carries both ``volume`` and ``return_mean`` arrays.

    vol_col: column name in df, or None when no volume data is available.
    ret_aligned: log-return Series aligned to df.index (NaN at first row).
    """
    idx = df.index
    v = df[vol_col].astype(float) if vol_col else None
    r = ret_aligned

    diffs = idx.to_series().diff().dropna()
    time_unit_min = max(1, int(diffs.median().total_seconds() / 60))
    counts_per_day = int(24 * 60 / time_unit_min)
    time_index = (idx.dayofweek * counts_per_day + idx.hour * (60 // time_unit_min) + idx.minute // time_unit_min)

    # ── Weekly (intraday × weekday) ────────────────────────────────────────────
    df_w = pd.DataFrame({"ti": time_index, "r": r.values}, index=idx)
    if v is not None:
        df_w["vol"] = v.values
    slot_range = range(counts_per_day * 7)
    ret_by_slot = df_w.groupby("ti")["r"].mean().reindex(slot_range)
    vol_by_slot = df_w.groupby("ti")["vol"].mean().reindex(slot_range) if v is not None else None
    weekly_days = []
    for d, label in enumerate(DAY_LABELS):
        start, end = d * counts_per_day, (d + 1) * counts_per_day
        weekly_days.append({
            "label": label,
            "slots": list(range(start, end)),
            "volume": [_vol(x) for x in vol_by_slot.iloc[start:end].values] if vol_by_slot is not None else [None] * counts_per_day,
            "return_mean": [_vol(x) for x in ret_by_slot.iloc[start:end].values],
        })
    slot_minutes = [t * time_unit_min for t in range(counts_per_day)]
    weekly = {
        "days": weekly_days,
        "counts_per_day": counts_per_day,
        "time_unit_min": time_unit_min,
        "time_labels": [f"{m // 60:02d}:{m % 60:02d}" for m in slot_minutes],
        "day_boundaries": [d * counts_per_day for d in range(8)],
    }

    # ── Monthly (weekday × week-of-month) ──────────────────────────────────────
    week_of_month = (idx.day - 1) // 7
    df_m = pd.DataFrame({"week": week_of_month, "dow": idx.dayofweek, "r": r.values}, index=idx)
    if v is not None:
        df_m["vol"] = v.values
    monthly_weeks = []
    for w in range(5):
        sub = df_m[df_m["week"] == w]
        if sub.empty:
            continue
        ret_agg = sub.groupby("dow")["r"].mean().reindex(range(7))
        vol_agg = sub.groupby("dow")["vol"].mean().reindex(range(7)) if v is not None else None
        if ret_agg.notna().any() or (vol_agg is not None and vol_agg.notna().any()):
            monthly_weeks.append({
                "label": f"Week {w + 1}",
                "days": list(range(7)),
                "volume": [_vol(x) for x in vol_agg.values] if vol_agg is not None else [None] * 7,
                "return_mean": [_vol(x) for x in ret_agg.values],
            })
    monthly = {"weeks": monthly_weeks, "day_labels": DAY_LABELS}

    # ── Day of month × year ────────────────────────────────────────────────────
    years = sorted(idx.year.unique().tolist())
    df_dom = pd.DataFrame({"day": idx.day, "year": idx.year, "r": r.values}, index=idx)
    if v is not None:
        df_dom["vol"] = v.values
    dom_series = []
    for yr in years:
        sub = df_dom[df_dom["year"] == yr]
        if sub.empty:
            continue
        ret_agg = sub.groupby("day")["r"].mean().reindex(range(1, 32))
        vol_agg = sub.groupby("day")["vol"].mean().reindex(range(1, 32)) if v is not None else None
        dom_series.append({
            "label": str(yr),
            "days": list(range(1, 32)),
            "volume": [_vol(x) for x in vol_agg.values] if vol_agg is not None else [None] * 31,
            "return_mean": [_vol(x) for x in ret_agg.values],
        })
    if len(years) > 1:
        ret_agg = df_dom.groupby("day")["r"].mean().reindex(range(1, 32))
        vol_agg = df_dom.groupby("day")["vol"].mean().reindex(range(1, 32)) if v is not None else None
        dom_series.insert(0, {
            "label": "avg",
            "days": list(range(1, 32)),
            "volume": [_vol(x) for x in vol_agg.values] if vol_agg is not None else [None] * 31,
            "return_mean": [_vol(x) for x in ret_agg.values],
        })
    day_of_month = {"series": dom_series}

    # ── Yearly (month × year) ──────────────────────────────────────────────────
    df_y = pd.DataFrame({"month": idx.month, "year": idx.year, "r": r.values}, index=idx)
    if v is not None:
        df_y["vol"] = v.values
    yearly_series = []
    for yr in years:
        sub = df_y[df_y["year"] == yr]
        if sub.empty:
            continue
        ret_agg = sub.groupby("month")["r"].mean().reindex(range(1, 13))
        vol_agg = sub.groupby("month")["vol"].mean().reindex(range(1, 13)) if v is not None else None
        yearly_series.append({
            "label": str(yr),
            "months": list(range(1, 13)),
            "volume": [_vol(x) for x in vol_agg.values] if vol_agg is not None else [None] * 12,
            "return_mean": [_vol(x) for x in ret_agg.values],
        })
    if len(years) > 1:
        ret_agg = df_y.groupby("month")["r"].mean().reindex(range(1, 13))
        vol_agg = df_y.groupby("month")["vol"].mean().reindex(range(1, 13)) if v is not None else None
        yearly_series.insert(0, {
            "label": "avg",
            "months": list(range(1, 13)),
            "volume": [_vol(x) for x in vol_agg.values] if vol_agg is not None else [None] * 12,
            "return_mean": [_vol(x) for x in ret_agg.values],
        })
    return {
        "weekly": weekly,
        "monthly": monthly,
        "day_of_month": day_of_month,
        "yearly": {"series": yearly_series, "month_labels": MONTH_LABELS},
    }


def _close_and_returns(df: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    """Extract close price series and log-returns from a DataFrame."""
    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", df.columns[-1])
    price = df[close_col].dropna()
    return price, _returns(price)


# ── Registered analyses (ordered fastest → slowest for progressive UX) ────────

@register("basic_stats")
def compute_basic_stats(df: pd.DataFrame) -> dict:
    """Row count, date range, basic OHLC stats. Fast, no heavy computation."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    price, r = _close_and_returns(df)
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


@register("return_dist")
def compute_return_dist(df: pd.DataFrame) -> dict:
    """Return distribution: summary stats + histogram vs Normal and Laplace fits."""
    price, r = _close_and_returns(df)
    if len(price) < 20:
        raise ValueError("Need at least 20 rows")
    mean_r, std_r = float(r.mean()), float(r.std())
    b_laplace = std_r / np.sqrt(2)
    counts, bin_edges = np.histogram(r, bins=HIST_BINS, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return {
        "stats": {
            "n": int(len(price)),
            "mean": mean_r,
            "std": std_r,
            "skewness": float(_skew(r)),
            "kurtosis": float(_kurtosis(r, fisher=False)),
            "hurst": _hurst(r),
        },
        "centers": centers.tolist(),
        "hist": [float(v) if v > 0 else None for v in counts.tolist()],
        "normal_pdf": norm.pdf(centers, loc=mean_r, scale=std_r).tolist(),
        "laplace_pdf": laplace.pdf(centers, loc=mean_r, scale=b_laplace).tolist(),
    }


@register("ccdf")
def compute_ccdf(df: pd.DataFrame) -> dict:
    """Fat-tail CCDF (log-log): P(|r| > x) vs |r|."""
    price, r = _close_and_returns(df)
    hist7, edges7 = np.histogram(np.abs(r), bins=HIST_BINS, density=True)
    cdf7 = np.cumsum(hist7 * np.diff(edges7))
    return {
        "x": edges7[1:].tolist(),
        "y": [float(v) if v > 0 else None for v in (1 - cdf7).tolist()],
    }


@register("diffusion")
def compute_diffusion(df: pd.DataFrame) -> dict:
    """Diffusion scaling: Var(lag) / Var(1) across multiple lags (log-log)."""
    price, _ = _close_and_returns(df)
    safe_periods = [int(p) for p in PERIODS if int(p) < len(price) // 2]
    vars_ = [float(np.var(price.diff(p).dropna())) for p in safe_periods]
    base = vars_[0] if vars_ and vars_[0] > 0 else 1.0
    return {"lags": safe_periods, "vars": [v / base for v in vars_]}


@register("acf")
def compute_acf(df: pd.DataFrame) -> dict:
    """Autocorrelation of returns and |returns| up to LAGS_ACF lags."""
    _, r = _close_and_returns(df)
    return {
        "returns": _acf_safe(r, LAGS_ACF),
        "abs_returns": _acf_safe(np.abs(r), LAGS_ACF),
    }


@register("vol_clustering")
def compute_vol_clustering(df: pd.DataFrame) -> dict:
    """ACF of |returns| up to LAGS_VOL lags — measures volatility clustering."""
    _, r = _close_and_returns(df)
    return {"values": _acf_safe(np.abs(r), LAGS_VOL)}


@register("qq")
def compute_qq(df: pd.DataFrame) -> dict:
    """QQ plot of returns vs Normal distribution."""
    _, r = _close_and_returns(df)
    (theoretical, sample), (slope, intercept, _) = probplot(r, dist="norm")
    step = max(1, len(theoretical) // 200)
    return {
        "points": [{"t": float(t), "s": float(s)} for t, s in zip(theoretical[::step], sample[::step])],
        "line": {"slope": float(slope), "intercept": float(intercept)},
    }


@register("exogenous_jump_tail")
def compute_exogenous_jump_tail(df: pd.DataFrame) -> dict:
    """Jump rate and tail quantiles of log-returns."""
    _, r = _close_and_returns(df)
    std = float(np.std(r))
    return {
        "jump_rate": float(np.mean(np.abs(r) > 3 * std)),
        "threshold_3sigma": 3 * std,
        "q99": float(np.quantile(r, 0.99)),
        "q999": float(np.quantile(r, 0.999)),
        "q001": float(np.quantile(r, 0.001)),
        "q01": float(np.quantile(r, 0.01)),
    }


@register("exogenous_cdf")
def compute_exogenous_cdf(df: pd.DataFrame) -> dict:
    """Empirical CDF of log-returns."""
    _, r = _close_and_returns(df)
    sorted_r = np.sort(r)
    n = len(sorted_r)
    idx = np.unique(np.linspace(0, n - 1, min(300, n)).astype(int))
    return {"x": sorted_r[idx].tolist(), "y": (idx / n).tolist()}


@register("exogenous_rolling_mean")
def compute_exogenous_rolling_mean(df: pd.DataFrame) -> dict:
    """Rolling mean of log-returns (window ≈ 5% of series length)."""
    _, r = _close_and_returns(df)
    window = max(10, len(r) // 20)
    rolling_vals = pd.Series(r).rolling(window).mean().dropna().values
    step = max(1, len(rolling_vals) // 400)
    sampled = rolling_vals[::step]
    return {"index": list(range(len(sampled))), "values": [float(v) for v in sampled], "window": window}


@register("exogenous_long_lag_acf")
def compute_exogenous_long_lag_acf(df: pd.DataFrame) -> dict:
    """ACF of log-returns up to 200 lags."""
    _, r = _close_and_returns(df)
    nlags = min(200, len(r) // 2 - 1)
    acf_vals = _acf_safe(r, nlags)
    return {
        "lags": list(range(len(acf_vals))),
        "values": acf_vals,
        "highlights": {str(lg): float(acf_vals[lg]) for lg in [10, 20, 30, 40, 50] if lg < len(acf_vals)},
    }


@register("exogenous_seasonality")
def compute_exogenous_seasonality(df: pd.DataFrame) -> dict:
    """Intraday and multi-scale seasonality patterns (slowest analysis)."""
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        raise ValueError(f"Cannot parse datetime index: {e}") from e
    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", df.columns[-1])
    price = df[close_col].dropna()
    if len(price) < 10:
        raise ValueError("Need at least 10 rows")
    returns = np.log(price).diff().dropna()
    vol_col = col_map.get("tick_volume") or col_map.get("volume")
    hourly_mean = returns.groupby(returns.index.hour).mean()
    hourly_std = returns.groupby(returns.index.hour).std().fillna(0)
    if vol_col:
        hvol = df[vol_col].astype(float).groupby(df.index.hour).mean().reindex(range(24))
        hourly_vol_mean: list | None = [_vol(x) for x in hvol.values]
    else:
        hourly_vol_mean = None
    ret_aligned = pd.Series(np.nan, index=df.index)
    ret_aligned.update(np.log(df[close_col]).diff())
    seasonality = None
    if len(df.index.to_series().diff().dropna()) > 0:
        seasonality = _seasonality(df, vol_col, ret_aligned)
    return {
        "intraday": {
            "hours": hourly_mean.index.tolist(),
            "return_mean": [float(v) for v in hourly_mean.values],
            "return_std": [float(v) for v in hourly_std.values],
            "volume_mean": hourly_vol_mean,
        },
        "seasonality": seasonality,
    }


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_df_for_dataset(artifact_path: str) -> pd.DataFrame:
    """Load a dataset parquet file (or DDM tick directory) into a DataFrame."""
    from pathlib import Path
    import os
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
    full_path = store / artifact_path
    if full_path.is_dir():
        from data.parquet_reader import load_ddm_ticks
        return load_ddm_ticks(full_path, max_files=100)
    return pd.read_parquet(full_path)


def stream_for_dataset(artifact_path: str) -> Iterator[tuple[str, dict]]:
    """
    Load dataset once, then yield (name, result) for each registered analysis in order.
    Analyses run sequentially; the caller may persist each result as it arrives.
    Raises on load error (caller should handle).
    """
    df = load_df_for_dataset(artifact_path)
    for name, fn in CHARACTERISTIC_REGISTRY.items():
        try:
            result = fn(df)
        except Exception as e:
            result = {"error": str(e)}
        yield name, result


def compute_for_dataset(artifact_path: str) -> dict:
    """
    Load a dataset and run all registered analyses.
    Returns dict keyed by analysis name.
    Used by the MCP server and batch tooling; for progressive saves use stream_for_dataset.
    """
    try:
        results: dict[str, Any] = {}
        for name, result in stream_for_dataset(artifact_path):
            results[name] = result
        return results
    except Exception as e:
        return {"error": str(e)}
