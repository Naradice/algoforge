"""
Multi-symbol (portfolio) backtester — cross-sectional, rebalanced, volatility-targeted books.

The single-symbol engine (strategy/engine/backtest.py) decides one instrument at a time. Strategies whose
edge is *cross-sectional* (carry baskets, currency strength) need all symbols at once, so this module takes a
universe of daily bars and a list of signal sleeves, and produces one portfolio return series.

Definition:
{
  "universe": [
     {"symbol": "USDJPY", "artifact_path": "research/fx/d1/USDJPY.parquet",   # parquet with close (DatetimeIndex)
      "base": "USD", "quote": "JPY", "spread_pct": 0.00003}                  # spread_pct: cost per unit turnover
  ],
  "rates":   {"dir": "research/fx/rates", "series": {"USD": "IR3TIB01USM156N", ...}},   # for carry sleeves
  "sleeves": [
     {"type": "carry",    "weight": 0.65, "threshold": 0.005, "scale": 0.03, "cap": 1.5},
     {"type": "momentum", "weight": 0.35, "lookbacks": [60, 250]}               # currency-strength momentum
  ],
  "overlays": [
     {"type": "vol_guard", "cap": 1.0},                                         # shrink when book vol >> its 1y norm
     {"type": "series_jump", "path": "datasets/src_65/...parquet", "column": "value", "ratio": 1.3,
      "lookback": 20, "cut": 0.7}                                              # e.g. VIX/VIX[-20] > 1.3 -> size x0.3
  ],
  "risk": {"target_vol": 0.10, "rebalance": "M", "max_leverage": 5.0, "vol_window": 60}
}

Conventions: decisions use information up to day t, positions apply from t+1; carry accrual is earned daily on
the held position (annual rate / 252); costs are charged on turnover. Returns are fractions of equity per
instrument at `target_vol` each, averaged over the instruments available that day.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _store() -> Path:
    return Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))


def _load_close(path: str) -> pd.Series:
    p = Path(path)
    p = p if p.is_absolute() else _store() / p
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, index_col=0, parse_dates=True)
    df.columns = [str(c).lower() for c in df.columns]
    s = df["close"].astype(float)
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _load_rates(spec: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    d = Path(spec["dir"])
    d = d if d.is_absolute() else _store() / d
    cols = {}
    for ccy, sid in spec["series"].items():
        s = pd.read_csv(d / f"{sid}.csv", index_col=0, parse_dates=True).iloc[:, 0]
        cols[ccy] = pd.to_numeric(s, errors="coerce").dropna() / 100.0
    m = pd.DataFrame(cols).shift(1)                       # monthly value known only from the next month
    return m.reindex(m.index.union(index)).ffill().reindex(index).ffill().bfill()


def _rebalance(signal: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Hold the last decision of each period for the whole next period."""
    key = signal.index.to_period({"M": "M", "W": "W", "D": "D"}.get(freq, "M"))
    out = {}
    for c in signal.columns:
        last = signal[c].groupby(key).last()
        out[c] = pd.Series(last.shift(1).reindex(key).values, signal.index)
    return pd.DataFrame(out).fillna(0.0)


def run_portfolio_backtest(definition: dict) -> dict:
    uni = definition["universe"]
    risk = definition.get("risk", {})
    target = float(risk.get("target_vol", 0.10))
    max_lev = float(risk.get("max_leverage", 5.0))
    win = int(risk.get("vol_window", 60))
    freq = str(risk.get("rebalance", "M"))

    closes = pd.DataFrame({u["symbol"]: _load_close(u["artifact_path"]) for u in uni})
    closes = closes[closes.index.dayofweek < 5].sort_index()
    lr = np.log(closes).diff()
    avail = closes.notna()
    vol = lr.rolling(win, min_periods=20).std().shift(1)
    scale = (target / np.sqrt(252) / vol).clip(upper=max_lev).fillna(0.0)
    syms = [u["symbol"] for u in uni]
    base = {u["symbol"]: u.get("base") for u in uni}
    quote = {u["symbol"]: u.get("quote") for u in uni}

    # ---- sleeves -> desired direction/weight per symbol
    carry = None
    desired = pd.DataFrame(0.0, index=closes.index, columns=syms)
    weights_sum = 0.0
    for sl in definition.get("sleeves", []):
        w = float(sl.get("weight", 1.0))
        if sl["type"] == "carry":
            rates = _load_rates(definition["rates"], closes.index)
            carry = pd.DataFrame({s: rates[base[s]] - rates[quote[s]] for s in syms})
            thr, sc, cap = float(sl.get("threshold", 0.005)), float(sl.get("scale", 0.03)), float(sl.get("cap", 1.5))
            sig = (carry / sc).clip(-cap, cap).where(carry.abs() > thr, 0.0)
        elif sl["type"] == "momentum":
            ccys = sorted({c for c in list(base.values()) + list(quote.values()) if c})
            sig = pd.DataFrame(0.0, index=closes.index, columns=syms)
            for L in sl.get("lookbacks", [60, 250]):
                r = np.log(closes).diff(L).fillna(0.0)
                score = pd.DataFrame(0.0, index=closes.index, columns=ccys)
                for s in syms:
                    score[base[s]] += r[s]
                    score[quote[s]] -= r[s]
                sig += pd.DataFrame({s: np.sign(score[base[s]] - score[quote[s]]) for s in syms})
            sig = sig / max(len(sl.get("lookbacks", [60, 250])), 1)
        else:
            raise ValueError(f"unknown sleeve type {sl['type']!r}")
        desired += w * _rebalance(sig, sl.get("rebalance", "W" if sl["type"] == "momentum" else freq))
        weights_sum += w
    if weights_sum > 0:
        desired /= weights_sum

    if carry is None:
        carry = pd.DataFrame(0.0, index=closes.index, columns=syms)
    pos = desired * scale * avail
    spread = pd.Series({u["symbol"]: float(u.get("spread_pct", 0.0)) for u in uni})
    cost = pos.diff().abs().fillna(0.0) * spread
    pnl = pos * lr.fillna(0.0) + pos * carry / 252.0 - cost
    book = pnl.sum(axis=1) / avail.sum(axis=1).clip(lower=1)

    # ---- overlays on the aggregated book
    for ov in definition.get("overlays", []):
        if ov["type"] == "vol_guard":
            bv = book.rolling(20).std().shift(1)
            med = book.rolling(500, min_periods=120).std().shift(1)
            book = book * (float(ov.get("cap", 1.0)) * med / bv).clip(upper=1.0).fillna(1.0)
        elif ov["type"] == "series_jump":
            s = _load_series(ov["path"], ov.get("column", "value"), ov.get("indicator")).reindex(book.index).ffill()
            lb = int(ov.get("lookback", 20))
            jump = (s / s.shift(lb) > float(ov.get("ratio", 1.3))).astype(float).shift(1).fillna(0.0)
            book = book * (1.0 - float(ov.get("cut", 0.7)) * jump)
        else:
            raise ValueError(f"unknown overlay type {ov['type']!r}")

    book = book[avail.any(axis=1)]
    equity = (1.0 + book.fillna(0.0)).cumprod()
    return {"returns": book, "equity": equity, "metrics": portfolio_metrics(book),
            "contribution": (pnl.sum() / avail.sum().clip(lower=1))}


def _load_series(path: str, column: str, indicator: str | None) -> pd.Series:
    p = Path(path)
    p = p if p.is_absolute() else _store() / p
    df = pd.read_parquet(p) if p.suffix == ".parquet" or p.is_dir() else pd.read_csv(p, index_col=0, parse_dates=True)
    if indicator is not None and "indicator" in df.columns:
        df = df[df["indicator"] == indicator]
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def portfolio_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return {"sharpe": 0.0, "annual_return": 0.0, "annual_vol": 0.0, "max_drawdown": 0.0, "days": int(len(r))}
    cum = r.cumsum()
    def sr(x): return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 50 and x.std() > 0 else None
    yearly = r.groupby(r.index.year).sum()
    return {"sharpe": sr(r), "sharpe_2009_plus": sr(r["2009":]), "annual_return": float(r.mean() * 252),
            "annual_vol": float(r.std() * np.sqrt(252)), "max_drawdown": float((cum - cum.cummax()).min()),
            "days": int(len(r)), "yearly": {int(k): round(float(v), 4) for k, v in yearly.items()}}
