"""Interest-rate differential series for the `carry` indicator (FRED 3m interbank CSVs, 1-month lag)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_IDS = {"USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "GBP": "IR3TIB01GBM156N",
        "JPY": "IR3TIB01JPM156N", "AUD": "IR3TIB01AUM156N", "CHF": "IR3TIB01CHM156N"}
_JPY_PRE2002 = "IRSTCB01JPM156N"


def _series(rates_dir: Path, sid: str) -> pd.Series:
    s = pd.read_csv(rates_dir / f"{sid}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna() / 100.0


def rate_differential(index: pd.DatetimeIndex, base: str, quote: str, rates_dir: str | None = None) -> pd.Series:
    """Annual (base - quote) short-rate differential aligned to `index` (month-start values, lagged one month)."""
    d = Path(rates_dir) if rates_dir else Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts")) / "research" / "fx" / "rates"
    cols = {}
    for c in (base, quote):
        if c not in _IDS:
            raise ValueError(f"carry: unsupported currency {c!r}; supported: {sorted(_IDS)}")
        s = _series(d, _IDS[c])
        if c == "JPY":
            s = s.combine_first(_series(d, _JPY_PRE2002))
        cols[c] = s
    m = pd.DataFrame(cols).shift(1)
    idx = pd.DatetimeIndex(index).tz_localize(None) if getattr(index, "tz", None) else pd.DatetimeIndex(index)
    aligned = m.reindex(m.index.union(idx)).ffill().reindex(idx).bfill()
    return pd.Series((aligned[base] - aligned[quote]).values, index=index)
