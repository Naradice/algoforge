"""
MT5 swap collector — snapshots the broker's real swap (rollover) rates per symbol.

Read-only: it only calls ``initialize`` / ``symbol_info`` / ``symbol_info_tick``; it never logs in with
credentials, places orders or changes terminal state. MT5 offers no swap *history*, so history is built by
re-running this collection (e.g. daily via a CollectionJob ``schedule_cron``); each run appends one row per symbol.

Datasource config:
    {"symbols": ["USDJPY", "EURUSD"], "terminal_path": null}

Output parquet (appended): datetime (UTC), symbol, swap_long, swap_short, swap_mode, rollover_3days,
point, digits, bid, spread_points, swap_long_ann, swap_short_ann

swap_*_ann is the annualised swap as a fraction of price where it can be derived from the swap mode
(points -> points*point/price*365; "interest" mode -> percent p.a./100), else NaN (raw values are kept).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))
# MT5 SYMBOL_SWAP_MODE_*: 0 disabled, 1 points, 2 base ccy, 3 interest (% p.a.), 4 margin ccy, 5 deposit ccy,
# 6 interest (% of current price), 7 interest (% of open price)
_POINTS, _INTEREST, _PCT_CUR, _PCT_OPEN = 1, 3, 6, 7


@dataclass
class CollectResult:
    artifact_path: str
    row_count: int
    from_ts: datetime
    to_ts: datetime


def annualise(swap: float, mode: int, point: float, price: float) -> float:
    """Annual fraction of price earned (+) / paid (-) per unit position, NaN when not derivable."""
    if mode == _POINTS and price > 0:
        return swap * point / price * 365.0
    if mode in (_INTEREST, _PCT_CUR, _PCT_OPEN):
        return swap / 100.0
    return float("nan")


def snapshot_rows(symbols: list[str], mt5, now: datetime) -> list[dict]:
    rows = []
    for sym in symbols:
        info = mt5.symbol_info(sym)
        if info is None:
            continue
        tick = mt5.symbol_info_tick(sym)
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        mode = int(info.swap_mode)
        rows.append({
            "datetime": now, "symbol": sym, "swap_long": float(info.swap_long), "swap_short": float(info.swap_short),
            "swap_mode": mode, "rollover_3days": int(getattr(info, "swap_rollover3days", -1)),
            "point": float(info.point), "digits": int(info.digits), "bid": bid,
            "spread_points": int(getattr(info, "spread", 0)),
            "swap_long_ann": annualise(float(info.swap_long), mode, float(info.point), bid),
            "swap_short_ann": annualise(float(info.swap_short), mode, float(info.point), bid),
        })
    return rows


def collect(datasource_id: int, config: dict, mt5_module=None) -> CollectResult:
    symbols = config.get("symbols") or ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCHF",
                                        "EURJPY", "GBPJPY", "AUDJPY", "CHFJPY"]
    mt5 = mt5_module
    if mt5 is None:
        import MetaTrader5 as mt5  # type: ignore
    ok = mt5.initialize(path=config["terminal_path"]) if config.get("terminal_path") else mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        rows = snapshot_rows(symbols, mt5, datetime.now(timezone.utc))
    finally:
        mt5.shutdown()
    if not rows:
        raise RuntimeError(f"MT5 returned no symbol info for {symbols}")

    out_dir = ARTIFACT_STORE / "datasets" / f"src_{datasource_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "swaps.parquet"
    new = pd.DataFrame(rows)
    new["datetime"] = pd.to_datetime(new["datetime"], utc=True)
    df = pd.concat([pd.read_parquet(path), new], ignore_index=True) if path.exists() else new
    df = df.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    df.to_parquet(path)
    return CollectResult(artifact_path=f"datasets/src_{datasource_id}/swaps.parquet", row_count=len(df),
                         from_ts=df["datetime"].min().to_pydatetime(), to_ts=df["datetime"].max().to_pydatetime())
