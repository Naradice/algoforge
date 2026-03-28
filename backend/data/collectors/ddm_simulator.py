"""
DDM (Deterministic Dealer Model) simulator collector.

Adapted from stocknet/stocknet/datasets/simulator.py — DeterministicDealerModelV3.

Datasource config shape:
    {
        "num_agent": 50,
        "max_volatility": 0.02,
        "min_volatility": 0.01,
        "trade_unit": 0.001,
        "initial_price": 100.0,
        "spread": 1.0,
        "wma": 5,
        "dealer_sensitive_min": -3.5,
        "dealer_sensitive_max": -1.5,
        "tick_time": 0.001,
        "length": 50000,          # number of ticks to simulate
        "timeframe": "M1",        # resample tick data to this OHLC frame
        "seed": 42
    }
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))


@dataclass
class CollectResult:
    artifact_path: str
    row_count: int
    from_ts: datetime
    to_ts: datetime


# ---------------------------------------------------------------------------
# DDM V3 — copied from stocknet/stocknet/datasets/simulator.py
# ---------------------------------------------------------------------------


class _DDMv1:
    """Minimal base for DeterministicDealerModelV3."""

    def __init__(
        self,
        num_agent: int,
        max_volatility: float = 0.02,
        min_volatility: float = 0.01,
        trade_unit: float = 0.001,
        initial_price: float = 100.0,
        spread: float = 1.0,
        tick_time: float = 0.001,
        **kwargs,
    ) -> None:
        tendency = pd.Series([random.uniform(min_volatility, max_volatility) for _ in range(num_agent)], dtype=float)
        prices = pd.Series([random.uniform(initial_price, initial_price + spread) for _ in range(num_agent)], dtype=float)
        position_trends = pd.Series([random.choice([-1, 1]) for _ in range(num_agent)], dtype=int)
        self.agent_df = pd.concat([tendency, position_trends, prices], axis=1, keys=["tend", "position", "price"])
        self.spread = spread
        self.market_price = initial_price + spread
        self.trade_unit = trade_unit
        self.tick_time = 0.0
        self.tick_time_unit = tick_time
        self.price_history: list[float] = [self.market_price]
        self.tick_times: list[float] = [0.0]

    def advance_order_price(self) -> pd.Series:
        self.agent_df["price"] += self.agent_df["position"] * self.agent_df["tend"]
        return self.agent_df["price"]

    def _contruct(self) -> None:
        prices = self.advance_order_price()
        bid = prices[self.agent_df["position"] == -1]
        ask = prices[self.agent_df["position"] == 1]
        if bid.empty or ask.empty:
            return
        bid_val, ask_val = bid.max(), ask.min()
        if bid_val >= ask_val:
            self.market_price = (bid_val + ask_val) / 2
            # Flip position for matched agents
            bid_idx = bid[bid == bid_val].index
            ask_idx = ask[ask == ask_val].index
            self.agent_df.loc[bid_idx, "position"] *= -1
            self.agent_df.loc[ask_idx, "position"] *= -1
        self.price_history.append(self.market_price)
        self.tick_time += self.tick_time_unit
        self.tick_times.append(self.tick_time)


class DDMv3(_DDMv1):
    """DeterministicDealerModelV3 with trend-following feedback."""

    def __init__(
        self,
        num_agent: int,
        max_volatility: float = 0.02,
        min_volatility: float = 0.01,
        trade_unit: float = 0.001,
        initial_price: float = 100.0,
        spread: float = 1.0,
        tick_time: float = 0.001,
        dealer_sensitive: float | Iterable | None = None,
        wma: int | Iterable = 5,
        dealer_sensitive_min: float = -3.5,
        dealer_sensitive_max: float = -1.5,
        **kwargs,
    ) -> None:
        super().__init__(num_agent, max_volatility, min_volatility, trade_unit, initial_price, spread, tick_time)

        if dealer_sensitive is None:
            self.dealer_sensitive = np.array([random.uniform(dealer_sensitive_min, dealer_sensitive_max) for _ in range(num_agent)])
        elif isinstance(dealer_sensitive, (int, float)):
            self.dealer_sensitive = np.full(num_agent, float(dealer_sensitive))
        else:
            self.dealer_sensitive = np.asarray(dealer_sensitive)

        if isinstance(wma, int):
            self.weight_array = np.array([random.uniform(0, 1) for _ in range(wma)])
            self.wma = wma
        else:
            self.weight_array = np.asarray(wma)
            self.wma = len(self.weight_array)
        self._total_weight = self.weight_array.sum()

    def _wma_diff(self) -> float:
        if len(self.price_history) < self.wma + 1:
            return 0.0
        diffs = [self.price_history[-i] - self.price_history[-i - 1] for i in range(1, self.wma + 1)]
        return float(np.dot(self.weight_array, diffs) / self._total_weight)

    def advance_order_price(self) -> pd.Series:
        follow = self.dealer_sensitive * self._wma_diff()
        self.agent_df["price"] += self.agent_df["position"] * self.agent_df["tend"] + follow
        return self.agent_df["price"]

    def simulate(self, length: int) -> pd.DataFrame:
        """Simulate `length` ticks and return tick price DataFrame."""
        while len(self.price_history) < length:
            self._contruct()
        return pd.DataFrame({"price": self.price_history[:length]})


# ---------------------------------------------------------------------------
# Timeframe resampling
# ---------------------------------------------------------------------------

_PANDAS_OFFSET = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def _ticks_to_ohlc(prices: pd.Series, freq: str) -> pd.DataFrame:
    ohlc = prices.resample(freq).ohlc()
    ohlc.columns = ["open", "high", "low", "close"]
    ohlc["volume"] = prices.resample(freq).count()
    return ohlc.dropna()


def collect(datasource_id: int, config: dict) -> CollectResult:
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    length = int(config.get("length", 50_000))
    timeframe = config.get("timeframe", "M1")
    freq = _PANDAS_OFFSET.get(timeframe, "1min")

    model = DDMv3(
        num_agent=int(config.get("num_agent", 50)),
        max_volatility=float(config.get("max_volatility", 0.02)),
        min_volatility=float(config.get("min_volatility", 0.01)),
        trade_unit=float(config.get("trade_unit", 0.001)),
        initial_price=float(config.get("initial_price", 100.0)),
        spread=float(config.get("spread", 1.0)),
        tick_time=float(config.get("tick_time", 0.001)),
        wma=int(config.get("wma", 5)),
        dealer_sensitive_min=float(config.get("dealer_sensitive_min", -3.5)),
        dealer_sensitive_max=float(config.get("dealer_sensitive_max", -1.5)),
    )

    tick_df = model.simulate(length)

    # Assign synthetic timestamps starting at 2000-01-03 00:00 UTC (Monday)
    tick_interval_s = float(config.get("tick_time", 0.001))
    start = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
    times = pd.date_range(start=start, periods=len(tick_df), freq=f"{tick_interval_s}s")
    tick_series = pd.Series(tick_df["price"].values, index=times)

    ohlc = _ticks_to_ohlc(tick_series, freq)

    out_dir = ARTIFACT_STORE / "datasets" / f"src_{datasource_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_rel = f"datasets/src_{datasource_id}/ddm_{timeframe}_len{length}.parquet"
    ohlc.to_parquet(ARTIFACT_STORE / artifact_rel)

    return CollectResult(
        artifact_path=artifact_rel,
        row_count=len(ohlc),
        from_ts=ohlc.index[0].to_pydatetime(),
        to_ts=ohlc.index[-1].to_pydatetime(),
    )
