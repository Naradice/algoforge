"""
DDM (Deterministic Dealer Model) simulator collector.

Faithful port of stocknet/stocknet/datasets/simulator.py —
DeterministicDealerModelV1 and DeterministicDealerModelV3.

Time handling mirrors the original exactly:
  - Every step (trade OR no-trade) advances self.tick_time by
    tick_time_unit * noise_factor via _add_time_for_ticks().
  - When a trade fires, the current self.tick_time is recorded as the
    trade timestamp.
  - For fixed-length collection: run until tick_time >= total_seconds
    (where total_seconds = length * candle_seconds).
  - For endless collection: run indefinitely, flushing batch files.

Datasource config shape:
    {
        "model":              "v3",       # "v1" or "v3"
        "num_agent":          300,
        "max_volatility":     0.02,
        "min_volatility":     0.01,
        "trade_unit":         0.001,
        "initial_price":      100.0,
        "spread":             1.0,
        "tick_time":          0.001,      # seconds per market-check step
        "time_noise_method":  "exp",      # "uniform" or "exp"
        "max_noise_factor":   100,        # ceiling for the noise multiplier
        "wma":                5,          # V3 only
        "dealer_sensitive_min": -3.5,     # V3 only
        "dealer_sensitive_max": -1.5,     # V3 only
        "length":             1000,       # desired OHLC candle count
        "timeframe":          "M1",
        "seed":               42
    }
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("ddm_simulator")

ARTIFACT_STORE = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))

_CANDLE_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}

_PANDAS_OFFSET: dict[str, str] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

BATCH_TICKS = 10_000


@dataclass
class CollectResult:
    artifact_path: str
    row_count: int
    from_ts: datetime
    to_ts: datetime


# ---------------------------------------------------------------------------
# DDM V1 / V3 — faithful numpy port of stocknet/stocknet/datasets/simulator.py
# ---------------------------------------------------------------------------


class _DDMv1:
    """DeterministicDealerModelV1 (Takayasu et al.).

    Closely mirrors the original DeterministicDealerModelV1:
      - Agent positions, tendencies, and prices are numpy arrays (faster than
        the original pandas DataFrame, but otherwise identical logic).
      - Time advances via _add_time_for_ticks() on every step, exactly as in
        the original's simulate(total_seconds) loop.
      - Balanced 50/50 position initialisation (matches original bull_ratio=0.5).
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
        time_noise_method: str | None = "exp",
        max_noise_factor: int = 100,
        initial_positions: list | None = None,
        **kwargs,
    ) -> None:
        # Tendency values — rounded to the same precision as trade_unit,
        # matching the original's `tendency.round(decimal_num)` logic.
        tend_raw = [random.uniform(min_volatility, max_volatility) for _ in range(num_agent)]
        if trade_unit < 1:
            decimal_num = 0
            d = 0.1
            for _ in range(100):
                decimal_num += 1
                if trade_unit / d >= 1 - trade_unit:
                    break
                d *= 0.1
            self.tend = np.round(np.array(tend_raw, dtype=np.float64), decimal_num)
        else:
            self.tend = np.array(tend_raw, dtype=np.float64)

        self.agent_prices = np.array(
            [random.uniform(initial_price, initial_price + spread) for _ in range(num_agent)],
            dtype=np.float64,
        )

        if initial_positions is not None:
            self.position = np.array(initial_positions, dtype=np.int8)
        else:
            # Balanced 50/50 — matches original bull_ratio=0.5.
            n_long = num_agent // 2
            pos: list[int] = [1] * n_long + [-1] * (num_agent - n_long)
            random.shuffle(pos)
            self.position = np.array(pos, dtype=np.int8)

        self.spread = spread
        self.market_price: float = initial_price + spread
        self.trade_unit = trade_unit
        self.tick_time: float = 0.0
        self.tick_time_unit: float = tick_time
        self.price_history: list[float] = [self.market_price]

        # Time-noise function — mirrors original's __get_time_noise setup.
        if time_noise_method is None or time_noise_method == "uniform":
            self._max_noise_factor = max_noise_factor
            self.__get_time_noise = self._get_uniform_noise
        elif time_noise_method == "exp":
            self.__noise_factors = list(range(1, max_noise_factor + 1))
            self.__noise_weights = [1 / (i + 1) for i in range(max_noise_factor)]
            self.__get_time_noise = self._get_weighted_noise
        else:
            raise ValueError(f"Unknown time_noise_method: {time_noise_method!r}")

    # -- noise helpers (mirror original) ------------------------------------

    def _get_uniform_noise(self) -> float:
        return random.uniform(1, self._max_noise_factor)

    def _get_weighted_noise(self) -> float:
        return random.choices(self.__noise_factors, weights=self.__noise_weights)[0]

    # -- core simulation steps (mirror original) ----------------------------

    def advance_order_price(self) -> None:
        """Advance all agent prices by position × tendency (no-trade step)."""
        self.agent_prices += self.position * self.tend

    def _add_time_for_ticks(self) -> None:
        """Advance simulated time by tick_time_unit × noise (every step)."""
        self.tick_time += self.tick_time_unit * self.__get_time_noise()

    def _contruct(self) -> bool:
        """Attempt to match a trade. Returns True if a trade occurred."""
        long_mask = self.position == 1
        short_mask = self.position == -1
        if not long_mask.any() or not short_mask.any():
            return False

        long_prices = self.agent_prices[long_mask]
        short_prices = self.agent_prices[short_mask]

        ask_price = long_prices.max()
        bid_price = short_prices.min() + self.spread

        if ask_price >= bid_price:
            self.market_price = (
                (ask_price + bid_price) / 2 // self.trade_unit
            ) * self.trade_unit
            long_idx = np.where(long_mask)[0][np.argmax(long_prices)]
            short_idx = np.where(short_mask)[0][np.argmin(short_prices)]
            self.position[long_idx] = -1
            self.position[short_idx] = 1
            return True
        return False

    def _common_step(self) -> tuple[float | None, float]:
        """One simulation step: try to trade, else advance prices."""
        if self._contruct():
            self.price_history.append(self.market_price)
            return self.market_price, self.tick_time
        else:
            self.advance_order_price()
            return None, self.tick_time

    def simulate_stream(self, n_trades: int | None = None, total_seconds: float | None = None):
        """Yield (price, tick_time) for each trade.

        Mirrors the original simulate(total_seconds) loop:
          - _add_time_for_ticks() is called on EVERY step (trade and no-trade).
          - The trade timestamp is self.tick_time at the moment of the trade.

        Stop conditions (first one that triggers wins):
          - total_seconds: stop when self.tick_time >= total_seconds
          - n_trades: stop after exactly n_trades yields
          - Neither: run forever (endless mode)

        Raises RuntimeError if no trade occurs for 10M consecutive steps
        (indicates model divergence — only a risk with num_agent < 50).
        """
        import itertools

        keep = getattr(self, "wma", 0) + 2
        yielded = 0
        no_trade_streak = 0
        _MAX_NO_TRADE = 10_000_000

        for _ in itertools.count():
            self._add_time_for_ticks()

            if total_seconds is not None and self.tick_time >= total_seconds:
                return

            price, tick = self._common_step()
            if price is not None:
                if len(self.price_history) > keep:
                    del self.price_history[0]
                yield price, tick
                yielded += 1
                no_trade_streak = 0
                if n_trades is not None and yielded >= n_trades:
                    return
            else:
                no_trade_streak += 1
                if no_trade_streak >= _MAX_NO_TRADE:
                    raise RuntimeError(
                        f"DDM simulation stalled: no trade in {_MAX_NO_TRADE:,} steps. "
                        "Agent prices likely diverged — use num_agent ≥ 300."
                    )

    def simulate(self, n_trades: int) -> pd.DataFrame:
        """Convenience: run for n_trades and return a DataFrame of prices."""
        prices = [p for p, _ in self.simulate_stream(n_trades=n_trades)]
        return pd.DataFrame({"price": prices})


class DDMv3(_DDMv1):
    """DeterministicDealerModelV3 — adds WMA trend-following feedback."""

    def __init__(
        self,
        num_agent: int,
        max_volatility: float = 0.02,
        min_volatility: float = 0.01,
        trade_unit: float = 0.001,
        initial_price: float = 100.0,
        spread: float = 1.0,
        tick_time: float = 0.001,
        time_noise_method: str | None = "exp",
        max_noise_factor: int = 100,
        dealer_sensitive: float | Iterable | None = None,
        wma: int | Iterable = 5,
        dealer_sensitive_min: float = -3.5,
        dealer_sensitive_max: float = -1.5,
        **kwargs,
    ) -> None:
        super().__init__(
            num_agent, max_volatility, min_volatility, trade_unit,
            initial_price, spread, tick_time, time_noise_method, max_noise_factor,
        )
        self.price_history = [self.market_price]

        if dealer_sensitive is None:
            self.dealer_sensitive = np.array(
                [random.uniform(dealer_sensitive_min, dealer_sensitive_max) for _ in range(num_agent)]
            )
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

    def advance_order_price(self) -> None:
        """Advance agent prices with WMA trend-following feedback."""
        wma = self._wma_diff()
        follow = self.dealer_sensitive * wma
        self.agent_prices += self.position * self.tend + follow


# Public alias
DDMv1 = _DDMv1


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def _clear_artifact_dir(out_dir: Path) -> None:
    """Remove all partition subdirectories and _meta.json from out_dir.

    Called at the start of collect() so re-running a simulation for the same
    datasource ID never mixes stale data from a previous run.
    """
    if not out_dir.exists():
        return
    import shutil
    # Remove Hive partition subdirectories (year=YYYY/)
    for p in out_dir.iterdir():
        if p.is_dir() and p.name.startswith("year="):
            shutil.rmtree(p)
    # Also remove any legacy batch_*.parquet files from old layout
    for f in out_dir.glob("batch_*.parquet"):
        f.unlink()
    meta = out_dir / "_meta.json"
    if meta.exists():
        meta.unlink()


def _write_batch(
    out_dir: Path,
    batch_num: int,
    prices: list[float],
    tick_times: list[float],
    start_ts: pd.Timestamp,
    total_trades: int = 0,
) -> None:
    """Write one batch of tick prices into a Hive date-partitioned parquet file.

    tick_times are simulated seconds from the model. Timestamps are:
        start_ts + timedelta(seconds=tick_time)

    Output path: out_dir/year=YYYY/month=MM/day=DD/part-NNNNNN.parquet
    Partition key is derived from the first tick timestamp in the batch.
    """
    timestamps = start_ts + pd.to_timedelta(tick_times, unit="s")
    df = pd.DataFrame({"price": prices}, index=timestamps)
    df.index.name = "datetime"

    # Derive partition from the first tick's date
    first_ts = timestamps[0]
    part_dir = (
        out_dir
        / f"year={first_ts.year}"
        / f"month={first_ts.month:02d}"
        / f"day={first_ts.day:02d}"
    )
    part_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(part_dir / f"part-{batch_num:06d}.parquet")

    # Update live metadata so the UI can show progress during a running job.
    from_ts = timestamps[0]
    to_ts = timestamps[-1]
    meta = {
        "total_trades": total_trades,
        "from_ts": from_ts.isoformat(),
        "to_ts": to_ts.isoformat(),
        "batch_num": batch_num,
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta))


def _ticks_to_ohlc(tick_series: pd.Series, timeframe: str) -> pd.DataFrame:
    """Resample a UTC-indexed tick price series to OHLC candles."""
    freq = _PANDAS_OFFSET.get(timeframe, "1min")
    ohlc = tick_series.resample(freq).ohlc()
    ohlc.columns = ["open", "high", "low", "close"]
    ohlc["volume"] = tick_series.resample(freq).count()
    return ohlc.dropna()


# ---------------------------------------------------------------------------
# collect() — entry point called by arq_worker
# ---------------------------------------------------------------------------


def collect(datasource_id: int, config: dict) -> CollectResult:
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    _raw_length = config.get("length")
    endless = _raw_length in (None, "", 0)
    ohlc_length = None if endless else int(_raw_length)

    timeframe = config.get("timeframe", "M1")
    candle_secs = _CANDLE_SECONDS.get(timeframe, 60)
    total_seconds = None if endless else float(ohlc_length * candle_secs)

    model_version = config.get("model", "v3").lower()
    log.info(
        f"DDM collect: datasource={datasource_id} model={model_version} "
        f"endless={endless} total_seconds={total_seconds} timeframe={timeframe}"
    )

    common_kwargs = dict(
        num_agent=int(config.get("num_agent", 300)),
        max_volatility=float(config.get("max_volatility", 0.02)),
        min_volatility=float(config.get("min_volatility", 0.01)),
        trade_unit=float(config.get("trade_unit", 0.001)),
        initial_price=float(config.get("initial_price", 100.0)),
        spread=float(config.get("spread", 1.0)),
        tick_time=float(config.get("tick_time", 0.001)),
        time_noise_method=config.get("time_noise_method", "exp"),
        max_noise_factor=int(config.get("max_noise_factor", 100)),
    )
    if model_version == "v1":
        model: _DDMv1 = DDMv1(**common_kwargs)
    else:
        model = DDMv3(
            **common_kwargs,
            wma=int(config.get("wma", 5)),
            dealer_sensitive_min=float(config.get("dealer_sensitive_min", -3.5)),
            dealer_sensitive_max=float(config.get("dealer_sensitive_max", -1.5)),
        )

    out_dir = ARTIFACT_STORE / "datasets" / f"src_{datasource_id}" / "ddm_ticks"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_rel = f"datasets/src_{datasource_id}/ddm_ticks"

    # Clear stale files so pd.read_parquet(directory) never mixes old + new batches.
    _clear_artifact_dir(out_dir)

    start_ts = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")

    prices_buf: list[float] = []
    ticks_buf: list[float] = []
    trade_count = 0
    batch_num = 0

    for price, tick in model.simulate_stream(total_seconds=total_seconds):
        trade_count += 1
        prices_buf.append(price)
        ticks_buf.append(tick)

        if len(prices_buf) >= BATCH_TICKS:
            _write_batch(out_dir, batch_num, prices_buf, ticks_buf, start_ts, total_trades=trade_count)
            log.info(f"DDM batch {batch_num} written: {BATCH_TICKS} trades, total={trade_count}")
            prices_buf.clear()
            ticks_buf.clear()
            batch_num += 1

    # Flush remaining ticks (fixed mode always lands here; endless never does).
    if prices_buf:
        _write_batch(out_dir, batch_num, prices_buf, ticks_buf, start_ts, total_trades=trade_count)
        batch_num += 1

    log.info(f"DDM collect done: {trade_count} total trades, {batch_num} batches")

    from data.parquet_reader import load_ddm_ticks
    tick_df = load_ddm_ticks(out_dir, max_files=batch_num + 1)
    tick_series = tick_df["price"]
    ohlc = _ticks_to_ohlc(tick_series, timeframe)

    return CollectResult(
        artifact_path=artifact_rel,
        row_count=len(ohlc),
        from_ts=ohlc.index[0].to_pydatetime(),
        to_ts=ohlc.index[-1].to_pydatetime(),
    )


# ---------------------------------------------------------------------------
# Live metadata (read by arq_worker + preview endpoint during endless runs)
# ---------------------------------------------------------------------------


def read_meta(datasource_id: int) -> dict:
    meta_path = (
        ARTIFACT_STORE / "datasets" / f"src_{datasource_id}" / "ddm_ticks" / "_meta.json"
    )
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}
