"""
Preprocessing pipeline applied to OHLC DataFrames before windowing.

Uses finance_client.fprocess as the single source of truth for all indicator math —
the same library and adapter pattern used by the strategy execution engine
(backend/strategy/engine/indicators.py).  Any fix to an indicator formula in
finance_client propagates here automatically.

Column-name convention produced by this module (used by OHLCWindowDataset and
mirrored in the frontend getOutputCols() helper):

    sma        → sma_{period}
    ema        → ema_{period}
    rsi        → rsi_{period}
    macd       → macd, macd_signal, macd_hist
    bbands     → bb_upper_{period}, bb_mid_{period}, bb_lower_{period}, bb_width_{period}
    atr        → atr_{period}
    returns    → returns_{period}   (pure pandas — no fprocess equivalent)
    volatility → vol_{period}       (pure pandas — no fprocess equivalent)
    clustering → cluster_{n}
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from finance_client.fprocess.fprocess.idcprocess import (
    ATRProcess,
    BBANDProcess,
    EMAProcess,
    MAProcess,
    MACDProcess,
    RSIProcess,
)


# ── Indicator adapters ─────────────────────────────────────────────────────
# Each function mirrors the adapter logic in strategy/engine/indicators.py,
# but maps to this module's fixed output-column naming convention.

def _add_sma(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = int(cfg.get("period", 20))
    col = cfg.get("column", "close")
    out = f"sma_{p}"
    proc = MAProcess(key=out, window=p, column=col)
    df = proc.run(df)
    # MAProcess produces "{out}_MA" → rename to our convention
    return df.rename(columns={f"{out}_MA": out})


def _add_ema(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = int(cfg.get("period", 20))
    col = cfg.get("column", "close")
    proc = EMAProcess(key=f"ema_{p}", window=p, column=col)
    return proc.run(df)  # EMAProcess names output after key directly ✓


def _add_rsi(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = int(cfg.get("period", 14))
    col = cfg.get("column", "close")
    out = f"rsi_{p}"
    # RSIProcess uses ohlc_column_name[0] as price source
    proc = RSIProcess(key=out, window=p, ohlc_column_name=(col,))
    return proc.run(df)  # Produces "{out}_Gain", "{out}_Loss", "{out}" — only "{out}" is selected


def _add_macd(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fast = int(cfg.get("fast", 12))
    slow = int(cfg.get("slow", 26))
    sig = int(cfg.get("signal", 9))
    col = cfg.get("column", "close")
    proc = MACDProcess(key="macd", target_column=col, short_window=fast, long_window=slow, signal_window=sig)
    df = proc.run(df)
    # MACDProcess produces: "MACD" (hardcoded), "macd_Signal", "macd_S_EMA", "macd_L_EMA"
    df = df.rename(columns={"MACD": "macd", "macd_Signal": "macd_signal"})
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def _add_bbands(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = int(cfg.get("period", 20))
    std_mult = float(cfg.get("std", 2.0))
    col = cfg.get("column", "close")
    key = f"bb_{p}"
    proc = BBANDProcess(key=key, window=p, alpha=std_mult, target_column=col)
    df = proc.run(df)
    # BBANDProcess produces "{key}_UV", "{key}_MV", "{key}_LV", "{key}_Width"
    return df.rename(columns={
        f"{key}_UV":    f"bb_upper_{p}",
        f"{key}_MV":    f"bb_mid_{p}",
        f"{key}_LV":    f"bb_lower_{p}",
        f"{key}_Width": f"bb_width_{p}",
    })


def _add_atr(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = int(cfg.get("period", 14))
    out = f"atr_{p}"
    # DataFrames are always lowercased by dataset.py before this is called
    proc = ATRProcess(key=out, window=p, ohlc_column_name=("open", "high", "low", "close"))
    return proc.run(df)  # ATRProcess names output after key directly ✓


def _add_returns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = int(cfg.get("period", 1))
    col = cfg.get("column", "close")
    df[f"returns_{p}"] = df[col].pct_change(p)
    return df


def _add_volatility(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    p = int(cfg.get("period", 20))
    col = cfg.get("column", "close")
    df[f"vol_{p}"] = df[col].pct_change().rolling(p).std()
    return df


_INDICATOR_MAP = {
    "sma":        _add_sma,
    "ema":        _add_ema,
    "rsi":        _add_rsi,
    "macd":       _add_macd,
    "bbands":     _add_bbands,
    "atr":        _add_atr,
    "returns":    _add_returns,
    "volatility": _add_volatility,
}


def add_indicators(df: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    for cfg in configs:
        fn = _INDICATOR_MAP.get(cfg.get("type", "").lower())
        if fn is not None:
            df = fn(df, cfg)
    return df


# ── Clustering ─────────────────────────────────────────────────────────────

def add_clustering(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if not cfg.get("enabled", False):
        return df

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    k = int(cfg.get("n_clusters", 5))
    on_cols = [c for c in cfg.get("on_cols", ["close"]) if c in df.columns]
    if not on_cols:
        return df

    feat = df[on_cols].ffill().dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat.values)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)

    col = f"cluster_{k}"
    df[col] = np.nan
    df.loc[feat.index, col] = labels.astype(float)
    return df


# ── Entry point ────────────────────────────────────────────────────────────

def apply_preprocessing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return augmented DataFrame with indicator and cluster columns appended."""
    indicators = config.get("indicators", [])
    if indicators:
        df = add_indicators(df, indicators)

    clustering = config.get("clustering", {})
    if clustering:
        df = add_clustering(df, clustering)

    return df
