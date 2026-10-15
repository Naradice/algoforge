"""Throwaway: generate N synthetic DDM regimes that vary only in dealer_sensitive bias (delta),
concatenate them into one combined OHLC dataset with an artificial timestamp gap inserted between
every pair of adjacent regimes, and register it as a Dataset row.

Train/held-out split is by WHOLE REGIME (first N_TRAIN_REGIMES regimes in the file vs. the last
N_HELDOUT_REGIMES), not by window: OHLCWindowDataset's require_contiguous=True + its existing
gap-detection (_compute_gap_mask, which flags any row-to-row timestamp delta that deviates from
the file's inferred median stride) will automatically exclude every window that would straddle a
regime boundary, so no window ever sees ticks from two different delta values. split_mode=
"chronological" with val_split=N_HELDOUT_REGIMES/N_REGIMES then puts the entire held-out block on
the val side -- exact because every regime contributes the same row count.

Uses DDMv3.simulate(n_trades=...) (trade-count-bounded -- confirmed empirically ~30x faster than
simulate_stream()'s wall-clock-bounded loop: 50,000 trades in ~1.8s vs. simulate_stream needing
~400s to reach the equivalent 300,000 simulated-seconds span) and buckets fixed-size trade groups
into synthetic OHLC candles directly, instead of DDMv3.simulate_stream()+_ticks_to_ohlc()'s
real-elapsed-time resampling. This experiment doesn't need production DDM tick-timing semantics
(the delta signal lives in the per-trade price path, not in inter-trade timing noise) -- just a
fast, well-controlled OHLC proxy for a given generative delta. A synthetic M1-spaced DatetimeIndex
is assigned afterward purely so the existing gap-detection machinery has something to key off.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.collectors.ddm_simulator import DDMv3

N_REGIMES = 100
N_TRAIN_REGIMES = 80
N_HELDOUT_REGIMES = 20
CANDLES_PER_REGIME = 1000
TRADES_PER_CANDLE = 20
N_TRADES_PER_REGIME = CANDLES_PER_REGIME * TRADES_PER_CANDLE
NUM_AGENT = 300
WMA = 5
DELTA_LOW, DELTA_HIGH = -1.5, 0.0
GAP_SECONDS = 86400  # 1 day -- far outside the 60s M1 stride, unambiguous gap
SEED = 42
DATASET_NAME = "ddm_regime_delta_regression"


def simulate_regime(delta: float, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    model = DDMv3(
        num_agent=NUM_AGENT, max_volatility=0.02, min_volatility=0.01, wma=WMA,
        dealer_sensitive_min=-3.5 + delta, dealer_sensitive_max=-1.5 + delta,
    )
    df = model.simulate(n_trades=N_TRADES_PER_REGIME)
    return df["price"].values


def trades_to_ohlc(prices: np.ndarray) -> pd.DataFrame:
    n = (len(prices) // TRADES_PER_CANDLE) * TRADES_PER_CANDLE
    grouped = prices[:n].reshape(-1, TRADES_PER_CANDLE)
    return pd.DataFrame({
        "open": grouped[:, 0],
        "high": grouped.max(axis=1),
        "low": grouped.min(axis=1),
        "close": grouped[:, -1],
        "volume": float(TRADES_PER_CANDLE),
    })


def build_combined() -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(SEED)
    deltas = rng.uniform(DELTA_LOW, DELTA_HIGH, size=N_REGIMES)
    order = rng.permutation(N_REGIMES)
    regime_order = list(order[:N_TRAIN_REGIMES]) + list(order[N_TRAIN_REGIMES:])

    blocks: list[pd.DataFrame] = []
    regime_map: list[dict] = []
    cursor_ts = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
    row_cursor = 0
    for i, ridx in enumerate(regime_order):
        delta = float(deltas[ridx])
        prices = simulate_regime(delta, seed=SEED + int(ridx))
        ohlc = trades_to_ohlc(prices).iloc[:CANDLES_PER_REGIME].copy()
        assert len(ohlc) == CANDLES_PER_REGIME, f"regime {ridx} produced {len(ohlc)} candles, expected {CANDLES_PER_REGIME}"

        idx = cursor_ts + pd.to_timedelta(np.arange(len(ohlc)) * 60, unit="s")
        ohlc.index = idx
        ohlc.index.name = "datetime"
        ohlc["delta"] = delta

        split = "train" if i < N_TRAIN_REGIMES else "heldout"
        regime_map.append({
            "regime_idx": int(ridx), "delta": delta, "split": split,
            "start_row": row_cursor, "n_rows": len(ohlc),
            "start_ts": idx[0].isoformat(), "end_ts": idx[-1].isoformat(),
        })
        blocks.append(ohlc)
        row_cursor += len(ohlc)
        cursor_ts = idx[-1] + pd.Timedelta(seconds=GAP_SECONDS)

    combined = pd.concat(blocks)
    assert len(combined) == N_REGIMES * CANDLES_PER_REGIME
    return combined, regime_map


async def register(combined: pd.DataFrame, regime_map: list[dict]) -> int:
    import database
    from data.models import Dataset

    # Default matches CLAUDE.md's documented ARTIFACT_STORE_PATH default ("../artifacts",
    # relative to backend/) -- NOT "artifacts" (a plain fallback copied from a call site that
    # normally never hits its own default because ARTIFACT_STORE_PATH is always set by the time
    # it runs). Getting this wrong once already sent a real run's output to backend/artifacts/
    # instead of the actual store the running Celery worker resolves against.
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
    artifact_name = f"datasets/derived/{DATASET_NAME}.parquet"
    (store / artifact_name).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(store / artifact_name)

    sidecar = store / f"datasets/derived/{DATASET_NAME}_regime_map.json"
    sidecar.write_text(json.dumps({
        "n_regimes": N_REGIMES, "n_train_regimes": N_TRAIN_REGIMES,
        "n_heldout_regimes": N_HELDOUT_REGIMES, "candles_per_regime": CANDLES_PER_REGIME,
        "trades_per_candle": TRADES_PER_CANDLE, "num_agent": NUM_AGENT, "wma": WMA,
        "delta_low": DELTA_LOW, "delta_high": DELTA_HIGH, "gap_seconds": GAP_SECONDS,
        "seed": SEED, "regimes": regime_map,
    }, indent=2))

    async with database.async_session_factory() as db:
        ds = Dataset(
            datasource_id=None,
            name=DATASET_NAME,
            symbol="DDM-SYNTH",
            timeframe="M1",
            row_count=len(combined),
            artifact_path=artifact_name,
            from_ts=combined.index[0].to_pydatetime(),
            to_ts=combined.index[-1].to_pydatetime(),
            status="ready",
        )
        db.add(ds)
        await db.commit()
        await db.refresh(ds)
        print(f"Registered dataset id={ds.id} name={ds.name!r} row_count={ds.row_count}")
        print(f"Regime map written to {sidecar}")
        return ds.id


def main():
    print(f"Generating {N_REGIMES} regimes ({N_TRAIN_REGIMES} train / {N_HELDOUT_REGIMES} held-out), "
          f"{CANDLES_PER_REGIME} candles each, delta ~ U({DELTA_LOW}, {DELTA_HIGH})...")
    combined, regime_map = build_combined()
    print(f"Combined shape: {combined.shape}, total gap boundaries: {N_REGIMES - 1}")

    # Sanity check: every inserted boundary really is a large gap, and every within-regime
    # transition really is exactly 60s -- catches an off-by-one in the concatenation logic
    # before this ever reaches OHLCWindowDataset's gap-detection.
    deltas_sec = combined.index.to_series().diff().dt.total_seconds().values[1:]
    boundary_positions = {r["start_row"] for r in regime_map if r["start_row"] > 0}
    for i, d in enumerate(deltas_sec, start=1):
        if i in boundary_positions:
            assert d > 3000, f"expected a large gap at row {i}, got {d}s"
        else:
            assert d == 60.0, f"expected 60s stride at row {i}, got {d}s"
    print("Gap sanity check passed: all regime boundaries are >3000s, all in-regime steps are 60s.")

    dataset_id = asyncio.run(register(combined, regime_map))
    print(f"\nDone. dataset_id={dataset_id}")


if __name__ == "__main__":
    main()
