"""
DDM (Deterministic Dealer Model) simulator collector.

Adapted from stocknet/stocknet/datasets/simulator.py — DeterministicDealerModelV3.

Tick data is stored as the artifact. When previewing/downloading a dataset the
service layer resamples the tick series to the requested OHLC timeframe on demand.

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
        "tick_time": 1.0,         # simulated seconds between ticks
        "length": 1000,           # desired number of OUTPUT OHLC candles
        "timeframe": "M1",        # OHLC timeframe for display / default resample
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
    """Minimal base for DeterministicDealerModelV3.

    Agent state is stored as numpy arrays (not a pandas DataFrame) so the inner
    simulation loop runs with minimal overhead.
    """

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
        self.tend = np.array([random.uniform(min_volatility, max_volatility) for _ in range(num_agent)], dtype=np.float64)
        self.agent_prices = np.array([random.uniform(initial_price, initial_price + spread) for _ in range(num_agent)], dtype=np.float64)
        self.position = np.array([random.choice([-1, 1]) for _ in range(num_agent)], dtype=np.int8)
        self.spread = spread
        self.market_price: float = initial_price + spread
        self.trade_unit = trade_unit
        self.tick_time = 0.0
        self.tick_time_unit = tick_time
        self.price_history: list[float] = [self.market_price]
        self.tick_times: list[float] = [0.0]

    def advance_order_price(self) -> np.ndarray:
        self.agent_prices += self.position * self.tend
        return self.agent_prices

    def _contruct(self, iter_index: int) -> None:
        """Run one simulation iteration.

        ``iter_index`` is the current iteration number; the simulated time of
        this iteration is ``iter_index * tick_time_unit``.  A price is only
        recorded when a trade actually occurs, but the timestamp is based on
        the iteration count — not on how many trades have happened.
        """
        prices = self.advance_order_price()
        seller_mask = self.position == -1
        buyer_mask = self.position == 1
        if not seller_mask.any() or not buyer_mask.any():
            return
        bid_val = prices[seller_mask].max()
        ask_val = prices[buyer_mask].min()
        if bid_val >= ask_val:
            # A trade occurred — update market price and record it with the
            # wall-clock time of *this iteration*, not trade-count time.
            self.market_price = (bid_val + ask_val) / 2
            flip_seller = seller_mask & (prices == bid_val)
            flip_buyer = buyer_mask & (prices == ask_val)
            self.position[flip_seller] *= -1
            self.position[flip_buyer] *= -1
            self.price_history.append(self.market_price)
            sim_time = iter_index * self.tick_time_unit
            self.tick_times.append(sim_time)


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
        h = self.price_history
        diffs = [h[-i] - h[-i - 1] for i in range(1, self.wma + 1)]
        return float(np.dot(self.weight_array, diffs) / self._total_weight)

    def advance_order_price(self) -> np.ndarray:
        wma = self._wma_diff()
        # Clip feedback relative to market price to prevent runaway divergence.
        # Without this, prices diverge exponentially and the simulation hangs.
        if self.market_price and np.isfinite(self.market_price):
            cap = abs(self.market_price) * 0.005
            wma = float(np.clip(wma, -cap, cap))
        follow = self.dealer_sensitive * wma
        self.agent_prices += self.position * self.tend + follow
        return self.agent_prices

    def simulate(self, n_iters: int) -> pd.DataFrame:
        """Run exactly ``n_iters`` simulation iterations.

        Returns a DataFrame with columns ``price`` and ``sim_time`` (seconds)
        for every iteration that produced a trade.  The time is iteration-based
        so candles contain only trades that fell within that time window.
        """
        for i in range(n_iters):
            self._contruct(i)
        # price_history[0] / tick_times[0] are the initial seed values — skip them
        return pd.DataFrame({
            "price": self.price_history[1:],
            "sim_time": self.tick_times[1:],
        })


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

# Approximate wall-clock seconds per OHLC candle for each timeframe
_CANDLE_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
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

    # `length` = desired number of output OHLC candles
    ohlc_length = int(config.get("length", 1000))
    timeframe = config.get("timeframe", "M1")
    freq = _PANDAS_OFFSET.get(timeframe, "1min")
    tick_interval_s = float(config.get("tick_time", 1.0))

    candle_secs = _CANDLE_SECONDS.get(timeframe, 60)
    ticks_per_candle = max(2, int(candle_secs / max(tick_interval_s, 0.001)))
    tick_count = ohlc_length * ticks_per_candle

    model = DDMv3(
        num_agent=int(config.get("num_agent", 50)),
        max_volatility=float(config.get("max_volatility", 0.02)),
        min_volatility=float(config.get("min_volatility", 0.01)),
        trade_unit=float(config.get("trade_unit", 0.001)),
        initial_price=float(config.get("initial_price", 100.0)),
        spread=float(config.get("spread", 1.0)),
        tick_time=tick_interval_s,
        wma=int(config.get("wma", 5)),
        dealer_sensitive_min=float(config.get("dealer_sensitive_min", -3.5)),
        dealer_sensitive_max=float(config.get("dealer_sensitive_max", -1.5)),
    )

    tick_df = model.simulate(tick_count)

    # Build timestamps from the actual iteration-based sim_time of each trade.
    # Trades that share the same integer-second bucket are placed at that second.
    start = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
    times = [start + pd.Timedelta(seconds=float(t)) for t in tick_df["sim_time"]]
    tick_series = pd.Series(tick_df["price"].values, index=times)

    # Compute OHLC for row_count / from_ts / to_ts metadata only
    ohlc = _ticks_to_ohlc(tick_series, freq)

    out_dir = ARTIFACT_STORE / "datasets" / f"src_{datasource_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Store raw tick data so the dataset can be resampled to any timeframe later
    artifact_rel = f"datasets/src_{datasource_id}/ddm_ticks.parquet"
    tick_series.rename("price").to_frame().to_parquet(ARTIFACT_STORE / artifact_rel)

    return CollectResult(
        artifact_path=artifact_rel,
        row_count=len(ohlc),
        from_ts=ohlc.index[0].to_pydatetime(),
        to_ts=ohlc.index[-1].to_pydatetime(),
    )
