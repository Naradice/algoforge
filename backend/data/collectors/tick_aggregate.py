"""
Tick aggregate collector -- local tick exports -> fixed-interval microstructure bars.

Datasource config shape (stored in datasources.config):
    {
        "path": "L:/data/fx/Dukascopy/USDJPY/tick",   # directory of tick files (read locally by the worker)
        "format": "dukascopy",                       # Dukascopy "Gmt time,Ask,Bid,AskVolume,BidVolume" daily CSVs
        "symbol": "USDJPY",
        "from_ts": "2017-01-01",                     # optional, by file date
        "to_ts": null,                               # optional, by file date (inclusive)
        "bar": "1min"                                # pandas offset of the output bars
    }

Output columns per bar (only bars with at least one tick):
    open high low close   mid price (bid + ask) / 2
    spread                mean ask - bid
    tick_count            number of top-of-book updates
    volume                = tick_count (quote activity; FX has no traded volume at the top of book)
    bid_size ask_size     mean top-of-book volume on each side
    imbalance             mean (bid_size - ask_size) / (bid_size + ask_size) -- order-book imbalance
    ofi                   order-flow imbalance summed over the bar (Cont, Kukanov & Stoikov 2014):
                          e_n = q^b_n 1[b_n >= b_{n-1}] - q^b_{n-1} 1[b_n <= b_{n-1}]
                                - q^a_n 1[a_n <= a_{n-1}] + q^a_{n-1} 1[a_n >= a_{n-1}]
                          (carried across file boundaries)

Files are processed one at a time and appended with a ParquetWriter, so memory stays at one day's
ticks regardless of the date range.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))

_DUKA_NAME = re.compile(r"_Ticks_(\d{2})\.(\d{2})\.(\d{4})-")
COLUMNS = ["open", "high", "low", "close", "spread", "tick_count", "volume", "bid_size", "ask_size",
           "imbalance", "ofi"]


@dataclass
class CollectResult:
    artifact_path: str
    row_count: int
    from_ts: datetime
    to_ts: datetime


def _dukascopy_files(path: Path, from_ts, to_ts) -> list[tuple[pd.Timestamp, Path]]:
    out = []
    for p in path.iterdir():
        m = _DUKA_NAME.search(p.name)
        if not m or p.suffix.lower() != ".csv":
            continue
        day = pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if (from_ts is None or day >= from_ts) and (to_ts is None or day <= to_ts):
            out.append((day, p))
    return sorted(out)


def read_dukascopy_ticks(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    ts = df.pop("gmt time").astype(str)
    # Dukascopy exports are day-first (dd.mm.yyyy); some tools re-save them as ISO yyyy-mm-dd.
    fmt = "%d.%m.%Y %H:%M:%S.%f" if len(ts) and ts.iloc[0][2:3] == "." else "ISO8601"
    df.index = pd.to_datetime(ts, format=fmt)
    return df.rename(columns={"askvolume": "ask_size", "bidvolume": "bid_size"})[["bid", "ask", "bid_size", "ask_size"]]


def order_flow_imbalance(ticks: pd.DataFrame, prev: pd.Series | None = None) -> np.ndarray:
    """Per-tick OFI e_n; the first tick uses `prev` (the last tick of the previous file) or 0."""
    b, a = ticks["bid"].to_numpy(), ticks["ask"].to_numpy()
    qb, qa = ticks["bid_size"].to_numpy(), ticks["ask_size"].to_numpy()
    if prev is not None:
        b0, a0, qb0, qa0 = [np.r_[prev[k], x[:-1]] for k, x in
                            (("bid", b), ("ask", a), ("bid_size", qb), ("ask_size", qa))]
    else:
        b0, a0, qb0, qa0 = [np.r_[x[0], x[:-1]] for x in (b, a, qb, qa)]
    e = qb * (b >= b0) - qb0 * (b <= b0) - qa * (a <= a0) + qa0 * (a >= a0)
    if prev is None:
        e[0] = 0.0
    return e


def aggregate_ticks(ticks: pd.DataFrame, bar: str = "1min", prev: pd.Series | None = None) -> pd.DataFrame:
    t = ticks.sort_index()
    mid = (t["bid"] + t["ask"]) / 2
    tot = t["bid_size"] + t["ask_size"]
    frame = pd.DataFrame({
        "mid": mid, "spread": t["ask"] - t["bid"], "bid_size": t["bid_size"], "ask_size": t["ask_size"],
        "imbalance": np.where(tot > 0, (t["bid_size"] - t["ask_size"]) / tot.where(tot > 0, 1), 0.0),
        "ofi": order_flow_imbalance(t, prev),
    }, index=t.index)
    g = frame.groupby(frame.index.floor(bar))
    out = g["mid"].ohlc()
    out["spread"] = g["spread"].mean()
    out["tick_count"] = g["mid"].size().astype(np.float64)
    out["volume"] = out["tick_count"]
    for c in ("bid_size", "ask_size", "imbalance"):
        out[c] = g[c].mean()
    out["ofi"] = g["ofi"].sum()
    out.index.name = "datetime"
    return out[COLUMNS]


def collect(datasource_id: int, config: dict) -> CollectResult:
    import pyarrow as pa
    import pyarrow.parquet as pq

    fmt = config.get("format", "dukascopy")
    if fmt != "dukascopy":
        raise ValueError(f"unsupported tick format {fmt!r} (supported: dukascopy)")
    path = Path(config["path"])
    if not path.is_dir():
        raise ValueError(f"tick directory not found: {path}")
    bar = config.get("bar", "1min")
    from_ts = pd.Timestamp(config["from_ts"]) if config.get("from_ts") else None
    to_ts = pd.Timestamp(config["to_ts"]) if config.get("to_ts") else None
    files = _dukascopy_files(path, from_ts, to_ts)
    if not files:
        raise ValueError(f"no {fmt} tick files in {path} for {from_ts} .. {to_ts}")

    symbol = config.get("symbol", path.parent.name)
    artifact_rel = f"datasets/src_{datasource_id}/{symbol}_ticks_{bar}.parquet"
    out_path = ARTIFACT_STORE / artifact_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()                      # a re-run rebuilds from scratch, never mixes old batches

    writer, rows, first, last, prev = None, 0, None, None, None
    try:
        for _, p in files:
            ticks = read_dukascopy_ticks(p)
            if ticks.empty:
                continue
            bars = aggregate_ticks(ticks, bar, prev)
            prev = ticks.sort_index().iloc[-1]
            table = pa.Table.from_pandas(bars, preserve_index=True)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
            rows += len(bars)
            first = bars.index[0] if first is None else first
            last = bars.index[-1]
    finally:
        if writer is not None:
            writer.close()
    if rows == 0:
        raise ValueError(f"tick files in {path} contained no ticks")

    from data.artifact_store import upload as _upload
    _upload(out_path)
    return CollectResult(artifact_path=artifact_rel, row_count=rows,
                         from_ts=first.to_pydatetime().replace(tzinfo=timezone.utc),
                         to_ts=last.to_pydatetime().replace(tzinfo=timezone.utc))
