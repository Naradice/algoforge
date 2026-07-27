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

import itertools
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import pywt
import ruptures as rpt
from scipy.signal import welch
from scipy.stats import laplace, norm, probplot
from scipy.stats import skew as _skew
from scipy.stats import kurtosis as _kurtosis
from statsmodels.tsa.stattools import acf as _acf
from statsmodels.tsa.stattools import adfuller as _adfuller
from statsmodels.tsa.stattools import bds as _bds

AnalysisFn = Callable[[pd.DataFrame], dict[str, Any]]

CHARACTERISTIC_REGISTRY: dict[str, AnalysisFn] = {}

# ── Config constants ──────────────────────────────────────────────────────────
LAGS_ACF = 50
LAGS_VOL = 100
CCDF_POINTS = 200
HIST_BINS = 100
PERIODS = [1, 2, 5, 10, 20, 50, 100, 200]

# Cap for O(n^2)-ish analyses (sample entropy, BDS test, changepoint detection) so they
# stay fast on tick-level datasets. Deterministic stride decimation preserves temporal order.
MAX_ANALYSIS_N = 5000


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
    """Hurst exponent via detrended fluctuation analysis (DFA).

    Cumulative sum of the demeaned series is split into boxes of size `s`; each box is
    linearly detrended and the RMS residual fluctuation F(s) is averaged. The slope of
    log(F(s)) vs log(s) is the Hurst exponent. More robust to non-stationary drift than a
    plain variance-scaling fit.
    """
    n = len(arr)
    if n < 32:
        return float("nan")
    profile = np.cumsum(arr - np.mean(arr))
    max_box = n // 4
    min_box = 4
    if max_box <= min_box:
        return float("nan")
    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 8).astype(int))
    box_sizes = box_sizes[box_sizes >= min_box]
    if len(box_sizes) < 2:
        return float("nan")

    fluctuations = []
    for s in box_sizes:
        n_boxes = n // s
        if n_boxes < 1:
            continue
        segments = profile[: n_boxes * s].reshape(n_boxes, s)
        x = np.arange(s)
        rms = []
        for seg in segments:
            coeffs = np.polyfit(x, seg, 1)
            trend = np.polyval(coeffs, x)
            rms.append(np.sqrt(np.mean((seg - trend) ** 2)))
        fluctuations.append(np.mean(rms))

    valid = [(s, f) for s, f in zip(box_sizes, fluctuations) if f > 0]
    if len(valid) < 2:
        return float("nan")
    sizes, fs = zip(*valid)
    poly = np.polyfit(np.log(sizes), np.log(fs), 1)
    return float(poly[0])


def _hurst_label(h: float) -> str:
    if np.isnan(h):
        return "undetermined"
    if h > 0.55:
        return "trending"
    if h < 0.45:
        return "mean-reverting"
    return "random walk"


def _decimate(arr: np.ndarray, max_n: int = MAX_ANALYSIS_N) -> tuple[np.ndarray, bool]:
    """Deterministic stride decimation to cap O(n^2)-ish analyses. Preserves temporal order."""
    n = len(arr)
    if n <= max_n:
        return arr, False
    stride = -(-n // max_n)  # ceiling division so len(arr[::stride]) <= max_n
    return arr[::stride], True


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


# ── Structure / complexity helpers ─────────────────────────────────────────────

def _spectral_entropy_from_psd(psd: np.ndarray) -> float:
    """Shannon entropy of a normalised power spectrum, scaled to [0, 1]."""
    total = psd.sum()
    if total <= 0:
        return float("nan")
    probs = psd[psd > 0] / total
    if len(probs) < 2:
        return float("nan")
    return float(-np.sum(probs * np.log2(probs)) / np.log2(len(probs)))


def _permutation_entropy(arr: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Bandt–Pompe permutation entropy, normalised to [0, 1] by log2(order!)."""
    n = len(arr)
    span = delay * (order - 1)
    if n <= span:
        return float("nan")
    patterns = list(itertools.permutations(range(order)))
    pattern_idx = {p: i for i, p in enumerate(patterns)}
    counts = np.zeros(len(patterns))
    for i in range(n - span):
        window = arr[i : i + span + 1 : delay]
        counts[pattern_idx[tuple(np.argsort(window))]] += 1
    counts = counts[counts > 0]
    if len(counts) < 2:
        return float("nan")
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)) / np.log2(len(patterns)))


def _sample_entropy(arr: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """SampEn(m, r): -log(A/B), A/B = count of (m+1)/m-length template matches within
    tolerance r (Chebyshev distance), each unordered pair counted once."""
    n = len(arr)
    if r is None:
        r = 0.2 * float(np.std(arr))
    if r <= 0 or n <= m + 2:
        return float("nan")

    def _match_count(mm: int) -> int:
        templates = np.array([arr[i : i + mm] for i in range(n - mm + 1)])
        total = 0
        for i in range(len(templates) - 1):
            dist = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            total += int(np.sum(dist <= r))
        return total

    b = _match_count(m)
    a = _match_count(m + 1)
    if b == 0 or a == 0:
        return float("nan")
    return float(-np.log(a / b))


# ── Registered analyses — long-term structure / complexity ────────────────────

@register("long_range_dependence")
def compute_long_range_dependence(df: pd.DataFrame) -> dict:
    """Hurst exponent (DFA), trending/mean-reverting/random-walk label, ADF stationarity,
    and effective memory length (first ACF lag to drop below the significance band)."""
    _, r = _close_and_returns(df)
    if len(r) < 32:
        raise ValueError("Need at least 32 returns")
    # adfuller's autolag search cost scales with both n and the data's own autocorrelation
    # structure (not just a clean O(n) or O(n^2)) -- on a 2M-row highly-regular series (e.g.
    # temporal XOR) it was observed to run for 80+ minutes before the worker got OOM-killed,
    # while an equally-sized but differently-structured series (LFSR) finished in seconds.
    # Decimate like every other expensive analysis in this module rather than trust autolag
    # to stay fast on arbitrary data.
    r, downsampled = _decimate(r)
    h = _hurst(r)
    nlags = min(200, len(r) // 2 - 1)
    acf_vals = _acf_safe(r, nlags)
    band = 2 / np.sqrt(len(r))
    memory_length = next(
        (lag for lag in range(1, len(acf_vals)) if abs(acf_vals[lag]) < band),
        len(acf_vals) - 1,
    )
    try:
        adf_stat, adf_pvalue = _adfuller(r)[:2]
    except Exception:
        adf_stat, adf_pvalue = float("nan"), float("nan")
    return {
        "hurst": h,
        "interpretation": _hurst_label(h),
        "memory_length": int(memory_length),
        "acf_significance_band": float(band),
        "acf_values": acf_vals,
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "n_used": int(len(r)),
        "downsampled": downsampled,
    }


@register("spectral_periodicity")
def compute_spectral_periodicity(df: pd.DataFrame) -> dict:
    """Welch power spectral density of returns: dominant period, periodicity strength
    (peak/mean power), low/mid/high frequency-band energy split, and spectral entropy."""
    _, r = _close_and_returns(df)
    if len(r) < 32:
        raise ValueError("Need at least 32 returns")
    nperseg = min(256, len(r))
    freqs, psd = welch(r, nperseg=nperseg)
    freqs, psd = freqs[1:], psd[1:]  # drop DC component
    if len(psd) == 0:
        raise ValueError("Insufficient data for spectral analysis")
    peak_idx = int(np.argmax(psd))
    dominant_freq = float(freqs[peak_idx])
    thirds = np.array_split(psd, 3)
    band_energy_raw = {"low": float(thirds[0].sum()), "mid": float(thirds[1].sum()), "high": float(thirds[2].sum())}
    total = sum(band_energy_raw.values()) or 1.0
    return {
        "frequencies": freqs.tolist(),
        "psd": psd.tolist(),
        "dominant_frequency": dominant_freq,
        "dominant_period": float(1 / dominant_freq) if dominant_freq > 0 else None,
        "periodicity_strength": float(psd[peak_idx] / np.mean(psd)),
        "band_energy": {k: v / total for k, v in band_energy_raw.items()},
        "spectral_entropy": _spectral_entropy_from_psd(psd),
    }


@register("multiscale_wavelet")
def compute_multiscale_wavelet(df: pd.DataFrame) -> dict:
    """Wavelet decomposition (db4) of returns: energy fraction per scale, and a
    "flatness score" (entropy of the energy distribution) — flat means short-, medium-,
    and long-term fluctuations coexist; concentrated means one scale dominates."""
    _, r = _close_and_returns(df)
    if len(r) < 32:
        raise ValueError("Need at least 32 returns")
    wavelet = "db4"
    max_level = pywt.dwt_max_level(len(r), pywt.Wavelet(wavelet).dec_len)
    level = max(1, min(6, max_level))
    coeffs = pywt.wavedec(np.array(r, dtype=np.float64, copy=True), wavelet=wavelet, level=level)
    energies = np.array([float(np.sum(np.square(c))) for c in coeffs])
    total = energies.sum() or 1.0
    energy_fraction = (energies / total).tolist()
    probs = energies[energies > 0] / total
    flatness_score = (
        float(-np.sum(probs * np.log2(probs)) / np.log2(len(probs))) if len(probs) > 1 else float("nan")
    )
    labels = [f"approx_L{level}"] + [f"detail_L{level - i}" for i in range(level)]
    return {
        "wavelet": wavelet,
        "level": level,
        "labels": labels,
        "energy_fraction": energy_fraction,
        "flatness_score": flatness_score,
    }


@register("complexity_nonlinearity")
def compute_complexity_nonlinearity(df: pd.DataFrame) -> dict:
    """Permutation entropy, sample entropy, and the BDS nonlinearity test on returns.
    A low BDS p-value means dependence remains after removing linear structure — i.e.
    the series has nonlinear dynamics a linear model can't capture."""
    _, r = _close_and_returns(df)
    if len(r) < 32:
        raise ValueError("Need at least 32 returns")
    r_capped, downsampled = _decimate(r)
    permutation_entropy = _permutation_entropy(r_capped, order=3, delay=1)
    sample_entropy = _sample_entropy(r_capped, m=2)
    try:
        bds_stat, bds_pvalue = _bds(r_capped, max_dim=2)
        bds_statistic = float(np.atleast_1d(bds_stat)[-1])
        bds_pvalue = float(np.atleast_1d(bds_pvalue)[-1])
    except Exception:
        bds_statistic, bds_pvalue = float("nan"), float("nan")
    return {
        "permutation_entropy": permutation_entropy,
        "sample_entropy": sample_entropy,
        "bds_statistic": bds_statistic,
        "bds_pvalue": bds_pvalue,
        "nonlinear": (bool(bds_pvalue < 0.05) if not np.isnan(bds_pvalue) else None),
        "n_used": int(len(r_capped)),
        "downsampled": downsampled,
    }


@register("regime_changes")
def compute_regime_changes(df: pd.DataFrame) -> dict:
    """Changepoint detection (PELT, RBF cost, BIC-style penalty) on returns — counts
    how often the series shifts between statistical regimes."""
    _, r = _close_and_returns(df)
    if len(r) < 32:
        raise ValueError("Need at least 32 returns")
    r_capped, downsampled = _decimate(r)
    n = len(r_capped)
    variance = float(np.var(r_capped))
    # BIC-style penalty for the L2 (mean-shift) cost; the RBF cost's penalty isn't
    # variance-scaled and over-segments badly at this magnitude — L2 tuned empirically
    # against synthetic mean-shift and pure-noise series (see plan/tests).
    penalty = 30.0 * float(np.log(n)) * variance if variance > 0 else 1.0
    algo = rpt.Pelt(model="l2").fit(r_capped.reshape(-1, 1))
    breakpoints = algo.predict(pen=penalty)
    changepoints = [int(b) for b in breakpoints if b < n]
    segment_lengths = np.diff([0] + changepoints + [n]).tolist()

    # Downsampled series + changepoints scaled to the same index space, for charting
    # (mirrors the ~400-point sampling convention used elsewhere, e.g. compute_exogenous_rolling_mean).
    step = max(1, n // 400)
    series_values = r_capped[::step].tolist()
    changepoints_display = [round(cp / step) for cp in changepoints]

    return {
        "n_changepoints": len(changepoints),
        "changepoints": changepoints,
        "n_segments": len(segment_lengths),
        "avg_segment_length": float(np.mean(segment_lengths)) if segment_lengths else None,
        "n_used": n,
        "downsampled": downsampled,
        "series_values": series_values,
        "changepoints_display": changepoints_display,
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
