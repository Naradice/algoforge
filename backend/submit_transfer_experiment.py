"""DDM -> USDJPY synthetic-pretraining transfer experiment (priority (1) of the user's staged
research plan: condition A [USDJPY from scratch] vs condition B [DDM pretrain -> USDJPY
fine-tune], fine-tune budget/model/data/seeds fully matched).

Two phases, run as separate CLI invocations:

    python submit_transfer_experiment.py prepare-data
        Collects the USDJPY and DDM datasets needed (no Celery worker required -- calls each
        collector's collect() directly, then registers the resulting Dataset row, the same way
        celery_worker.py's run_collection_job would). Prints the resulting dataset ids; paste
        them into USDJPY_DATASET_ID / DDM_DATASET_ID below before running smoke/full.

    python submit_transfer_experiment.py smoke
        1 seed, small max_steps. Submits one condition-A run and one condition-B pretrain+
        fine-tune pair via the real TrainingRun/Celery pipeline (requires a `training`-queue
        Celery worker running -- see CLAUDE.md). Verifies warm_start_checkpoint actually loads
        (condition B's fine-tune run should NOT start from the same initial val_loss as
        condition A) before committing to the full run.

    python submit_transfer_experiment.py full [seed ...]
        N_SEEDS_FULL (default 3) condition-A runs + 1 pretrain + N_SEEDS_FULL condition-B
        fine-tune runs, at the real step budget.

Follows backend/submit_regime_training.py's pattern: same create_training_run + enqueue
("train_model", run.id) path the API router uses, so results show up in the real training
pipeline (TrainingRun/TrainingCheckpoint/model-compare), not a disconnected reimplementation.

Both DDM and USDJPY pretraining/fine-tuning use the SAME target task (close -> vol_{PERIOD},
the "return -> realized volatility" convention established across all five datasets in the
"Five Axes of Scaling" investigation, docs/research-seed-five-axes-of-scaling.md) via the
existing preprocessing.py "volatility" indicator + OHLCWindowDataset's tgt_feature_cols -- no
new dataset-prep code needed, just the right hyperparams.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Fill these in after `prepare-data` prints the registered dataset ids.
# ---------------------------------------------------------------------------
# dataset id=29 "USDJPY" (M1, 2,239,284 rows, 2016-08-31 -> 2022-08-31) -- a real, full-history
# dataset already registered in this project (the same one the "Five Axes of Scaling"
# investigation's usdjpy_1min_volatility.parquet derives from), NOT the ~1-month yfinance
# snippet `prepare-data` used to collect fresh (dataset id=51, 29,611 rows) -- yfinance's M1
# endpoint is provider-limited to recent history only; there was no need to hit it at all once
# this was found. Note: dataset 29's own derived sibling (id=30, "usdjpy_1min_volatility") has
# precomputed rv/log_rv columns, but their exact formula isn't documented anywhere in this repo
# and empirically isn't just a plain rolling-std-of-returns at any clean window (correlation
# ~0.91 at best against several tried, never 1.0) -- reusing an unverified legacy column would
# also break target-formula consistency with DDM's side. Using our own preprocessing
# "volatility" indicator (see BASE_HP below) on dataset 29's raw OHLC instead keeps the target
# definition IDENTICAL between DDM and USDJPY, which matters more for a fair transfer comparison
# than reusing whatever the original undocumented rv/log_rv computation was.
USDJPY_DATASET_ID: int | None = 29
DDM_DATASET_ID: int | None = 52
# Condition C's mixture pretrain dataset -- filled in by `prepare-mixture-data`.
MIXTURE_DATASET_ID: int | None = None
# Filled in by `prepare-data`/first submit -- the shared decoder_only MLModel both conditions'
# runs are created under (same architecture config = same warm-started weight shapes).
ML_MODEL_ID: int | None = None

VOL_PERIOD = 20  # bars; also fixes the tgt_feature_cols column name "vol_{VOL_PERIOD}"

# Only used by `prepare-data`'s (now unnecessary, kept only as a documented fallback -- see
# USDJPY_DATASET_ID's comment above) fresh yfinance collection path.
USDJPY_SYMBOL = "USDJPY=X"  # yfinance forex ticker format (plain "USDJPY" resolves to no data)
USDJPY_TIMEFRAME = "M1"

# Row cap applied to BOTH datasets via BASE_HP's max_rows (DDM's 300k-row dataset is unaffected;
# this caps USDJPY's 2.24M-row real history to a recent, still-substantial 1M-row / ~2-year
# slice instead of silently falling back to OHLCWindowDataset's own 50,000-row default -- see
# CLAUDE.md's "Always pass max_rows explicitly" standing lesson).
MAX_ROWS = 1_000_000

# Pretraining budget is explicitly NOT required to match condition A's data volume -- only the
# fine-tune side must match (user's own Phase 2 spec) -- picked independently, generous since
# it's "free".
#
# Generated as many SHORT independent runs concatenated together, NOT one long continuous run:
# a single DDMv3 trajectory long enough to cover DDM_PRETRAIN_ROWS candles (num_agent=300-500,
# tried directly against ddm_simulator.collect()'s wall-clock-bounded simulate_stream) reliably
# hit "DDM simulation stalled ... Agent prices likely diverged" -- WMA-feedback price divergence
# over a long single horizon, the same failure mode already on record in project memory ("DDM
# Simulator -- small agent count WMA divergence", "DDM is non-stationary over large data
# volumes"). generate_regime_datasets.py's established fix is exactly this: many short, freshly-
# seeded runs (divergence risk resets each run) concatenated with timestamp gaps, relying on
# OHLCWindowDataset's require_contiguous to keep windows from crossing a run boundary.
DDM_PRETRAIN_ROWS = 300_000
DDM_CANDLES_PER_RUN = 5_000
DDM_N_RUNS = DDM_PRETRAIN_ROWS // DDM_CANDLES_PER_RUN
DDM_TRADES_PER_CANDLE = 20
DDM_GAP_SECONDS = 86_400  # 1 day -- far outside the 60s M1 stride, unambiguous gap
DDM_NUM_AGENT = 500

# Condition C (priority (2)): DDM + Sine + Delay(Mackey-Glass) + XOR(temporal) + LFSR(8-bit),
# each contributing an equal share so the mixture's TOTAL row count equals DDM_PRETRAIN_ROWS --
# see _build_synthetic_mixture_data's docstring for why (isolating "more structural diversity"
# from "more data" is the whole point of comparing B vs C, per the user's own Phase 3/4 note).
MIXTURE_ROWS_PER_SOURCE = DDM_PRETRAIN_ROWS // 5  # 60_000 at current DDM_PRETRAIN_ROWS

SEEDS_FULL = [42, 43, 44]

# Fine-tune-side budget -- IDENTICAL across condition A and condition B's fine-tune runs. Start
# conservative; recalibrate after `smoke` reports real wall-clock-per-step on this machine.
FINETUNE_MAX_STEPS = 20_000
FINETUNE_VAL_EVERY_STEPS = 1_000
FINETUNE_EARLY_STOP_PATIENCE_CHECKS = 5

# Pretrain-side budget -- independent of the fine-tune budget above (see DDM_PRETRAIN_ROWS note).
PRETRAIN_MAX_STEPS = 40_000
PRETRAIN_VAL_EVERY_STEPS = 2_000

MODEL_CONFIG = {
    # Pre-LN + small LayerScale: the fix (see uncommitted backend/model_core/architectures/
    # decoder_only.py diff / attention_no_layernorm_diagnostic.py) for the collapse plain
    # post-LN causes on a volatility-like target that needs magnitude information preserved.
    "layernorm_mode": "pre",
    "layerscale_init": 1e-2,
    # "last" (causal pooling), not "mean" -- the Five Axes investigation found the attention/
    # order mechanism itself (not just distributional/pairwise statistics) is what separates
    # Transformer from a permutation-invariant baseline on this exact target; mean-pooling with
    # full attention would give that mechanism away for free instead of testing it.
    "pooling": "last",
    "d_model": 64,
    "nhead": 4,
    "num_layers": 4,
    "dim_feedforward": 256,
    "dropout": 0.1,
}

BASE_HP = {
    "obs_len": 60,
    "pred_len": 1,
    "feature_cols": ["close"],
    "tgt_feature_cols": [f"vol_{VOL_PERIOD}"],
    "preprocessing": {"indicators": [{"type": "volatility", "period": VOL_PERIOD, "column": "close"}]},
    # zscore (not "none"): DDM's and USDJPY's raw vol_{PERIOD} scales differ by construction
    # (different price/tick conventions) -- standardizing per-dataset puts warm-started weights
    # on a comparable target scale, the same rationale as dataset.py's "returns_zscore" input
    # transform. Without this, most of any transfer effect (or lack of one) could just be a
    # trivial output-scale mismatch the fine-tune run has to relearn regardless of pretraining.
    "normalize": "zscore",
    # "regime_controlled" (not "chronological"): the first full run on the real 1M-row USDJPY
    # dataset (TrainingRuns 1432-1438) showed every condition-A seed's best val_loss landing at
    # the very first checkpoint then degrading monotonically -- diagnosed as a real train/val
    # regime shift, not noise (chronological split puts validation mostly in 2022's USDJPY
    # volatility surge, a different regime than the 2019-2021 training period). This sanity-check
    # rerun (user-requested, see conversation) stratifies train/val by the target's own value
    # distribution instead of by time (OHLCWindowDataset's regime_controlled mode -- deciles of
    # each window's target level, split within each decile), so a genuine transfer effect isn't
    # confounded with "which condition happens to handle the 2022 regime better." Trade-off
    # (documented on the mode itself in dataset.py): no longer a walk-forward evaluation, an
    # in-distribution one -- appropriate for isolating the pretraining question, not a claim this
    # is now a deployable forecast.
    "split_mode": "regime_controlled",
    "require_contiguous": True,
    "max_rows": MAX_ROWS,
    "val_split": 0.2,
    "batch_size": 64,
    "disable_lr_scheduler": True,
    # NOTE: vol_{PERIOD}'s first (PERIOD-1) rows are NaN (rolling std warm-up) -- preprocessing.py
    # doesn't drop them and OHLCWindowDataset's require_contiguous gap-mask is timestamp-gap-based,
    # not NaN-based, so a handful of early windows can carry a NaN target. Verified directly (see
    # conversation) that with max_rows=1,000,000 on both the USDJPY and DDM datasets used here,
    # zero windows in either the train or val split actually carry a NaN target -- the leading
    # NaN block is a negligible ~19 rows out of hundreds of thousands and gets diluted away.
}


def _finetune_hp(seed: int, warm_start_checkpoint: str | None) -> dict:
    hp = {
        **BASE_HP,
        "seed": seed,
        "max_steps": FINETUNE_MAX_STEPS,
        "val_every_steps": FINETUNE_VAL_EVERY_STEPS,
        "early_stop_patience_checks": FINETUNE_EARLY_STOP_PATIENCE_CHECKS,
    }
    if warm_start_checkpoint:
        hp["warm_start_checkpoint"] = warm_start_checkpoint
    return hp


def _pretrain_hp(seed: int, max_steps: int, val_every_steps: int) -> dict:
    return {
        **BASE_HP,
        "seed": seed,
        "max_steps": max_steps,
        "val_every_steps": val_every_steps,
        # No early stopping on the pretrain side -- it's not the comparison being measured, and
        # early-stopping the representation-building phase early is not obviously desirable here.
    }


def _simulate_ddm_segment(total_rows: int, candles_per_run: int = DDM_CANDLES_PER_RUN,
                           cursor_ts=None, seed_offset: int = 1000):
    """n_runs = total_rows // candles_per_run independent, freshly-seeded DDMv3 (v3_shock
    params) runs of candles_per_run candles each, concatenated with a timestamp gap between
    every pair -- see DDM_PRETRAIN_ROWS's comment for why not one long run. Returns
    (combined_df, from_ts, to_ts). Mirrors generate_regime_datasets.py's simulate/trades_to_ohlc/
    build_combined, which established this exact pattern for the same reason (avoiding DDMv3's
    long-horizon WMA divergence). seed_offset lets condition C's smaller DDM slice use a
    disjoint seed range from condition B's full-size pretrain, so they're not literally the same
    trajectories truncated."""
    import numpy as np
    import pandas as pd
    from data.collectors.ddm_simulator import DDMv3

    n_runs = total_rows // candles_per_run
    n_trades_per_run = candles_per_run * DDM_TRADES_PER_CANDLE
    cursor_ts = cursor_ts if cursor_ts is not None else pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
    blocks = []
    for run_idx in range(n_runs):
        seed = seed_offset + run_idx
        np.random.seed(seed)
        model = DDMv3(
            num_agent=DDM_NUM_AGENT, max_volatility=0.02, min_volatility=0.01, wma=5,
            dealer_sensitive_min=-3.5, dealer_sensitive_max=-1.5,
            exogenous_shock_probability=0.0015, exogenous_shock_size=0.3,
        )
        prices = model.simulate(n_trades=n_trades_per_run)["price"].values

        n = (len(prices) // DDM_TRADES_PER_CANDLE) * DDM_TRADES_PER_CANDLE
        grouped = prices[:n].reshape(-1, DDM_TRADES_PER_CANDLE)
        ohlc = pd.DataFrame({
            "open": grouped[:, 0], "high": grouped.max(axis=1), "low": grouped.min(axis=1),
            "close": grouped[:, -1], "volume": float(DDM_TRADES_PER_CANDLE),
        }).iloc[:candles_per_run]

        idx = cursor_ts + pd.to_timedelta(np.arange(len(ohlc)) * 60, unit="s")
        ohlc.index = idx
        ohlc.index.name = "datetime"
        blocks.append(ohlc)
        cursor_ts = idx[-1] + pd.Timedelta(seconds=DDM_GAP_SECONDS)
        print(f"  DDM segment run {run_idx + 1}/{n_runs}: {len(ohlc)} candles")

    combined = pd.concat(blocks)
    return combined, combined.index[-1] + pd.Timedelta(seconds=DDM_GAP_SECONDS)


def _simulate_ddm_pretrain_data():
    """DDM-only pretrain dataset (DDM_PRETRAIN_ROWS candles) -- see _simulate_ddm_segment."""
    combined, _next_cursor = _simulate_ddm_segment(DDM_PRETRAIN_ROWS)
    return combined, combined.index[0].to_pydatetime(), combined.index[-1].to_pydatetime()


def _generate_synthetic_segment(function: str, length: int, seed: int, cursor_ts, **extra_config):
    """One synthetic_function.py series (sine/delay/xor/lfsr), as a flat-candle OHLC DataFrame
    re-indexed to start at cursor_ts -- calls the collector's own _generate_series directly
    (not collect(), which writes its own standalone file/datasource-keyed artifact; here we only
    want the raw values to fold into one combined mixture dataset, same reasoning as
    _simulate_ddm_segment building its own OHLC rather than going through ddm_simulator.collect()).
    Returns (df, next_cursor_ts)."""
    import numpy as np
    import pandas as pd
    from data.collectors.synthetic_function import _generate_series

    period = float(extra_config.get("period", 50))
    amplitude = float(extra_config.get("amplitude", 1.0))
    freq_ratio = float(extra_config.get("freq_ratio", 5))
    tau = float(extra_config.get("tau", 17))
    lfsr_bits = int(extra_config.get("lfsr_bits", 8))
    base_price = float(extra_config.get("base_price", 100.0))

    values = base_price + _generate_series(
        function, length, period, amplitude, freq_ratio, tau=tau, lfsr_bits=lfsr_bits, seed=seed
    )
    idx = cursor_ts + pd.to_timedelta(np.arange(length) * 60, unit="s")
    df = pd.DataFrame({
        "open": values, "high": values, "low": values, "close": values,
        "volume": np.ones(length),
    }, index=idx)
    df.index.name = "datetime"
    next_cursor = idx[-1] + pd.Timedelta(seconds=DDM_GAP_SECONDS)
    print(f"  synthetic segment {function!r}: {length} candles")
    return df, next_cursor


def _build_synthetic_mixture_data():
    """Sine + Delay(Mackey-Glass) + XOR(temporal) + LFSR(8-bit) + DDM(v3_shock), each
    MIXTURE_ROWS_PER_SOURCE candles, concatenated with a timestamp gap between every segment --
    condition C's pretraining data. Total rows == DDM_PRETRAIN_ROWS (condition B's pretrain
    volume) by construction: the whole point of comparing B vs C is "same total pretraining
    volume, different composition" (per the user's own Phase 3/4 methodology note --
    separate 'more DDM data' from 'more structural diversity'), not "C also has more data than
    B." Returns (combined_df, from_ts, to_ts)."""
    import pandas as pd

    cursor_ts = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
    blocks = []

    sine_df, cursor_ts = _generate_synthetic_segment(
        "sine", MIXTURE_ROWS_PER_SOURCE, seed=2001, cursor_ts=cursor_ts, period=50, amplitude=1.0
    )
    blocks.append(sine_df)

    delay_df, cursor_ts = _generate_synthetic_segment(
        "delay", MIXTURE_ROWS_PER_SOURCE, seed=2002, cursor_ts=cursor_ts, tau=17
    )
    blocks.append(delay_df)

    xor_df, cursor_ts = _generate_synthetic_segment(
        "xor", MIXTURE_ROWS_PER_SOURCE, seed=2003, cursor_ts=cursor_ts, amplitude=1.0
    )
    blocks.append(xor_df)

    lfsr_df, cursor_ts = _generate_synthetic_segment(
        "lfsr", MIXTURE_ROWS_PER_SOURCE, seed=2004, cursor_ts=cursor_ts, lfsr_bits=8, amplitude=1.0
    )
    blocks.append(lfsr_df)

    ddm_df, cursor_ts = _simulate_ddm_segment(
        MIXTURE_ROWS_PER_SOURCE, candles_per_run=DDM_CANDLES_PER_RUN, cursor_ts=cursor_ts, seed_offset=3000
    )
    blocks.append(ddm_df)

    combined = pd.concat(blocks)
    return combined, combined.index[0].to_pydatetime(), combined.index[-1].to_pydatetime()


def _read_cached_yfinance_ohlc(symbol: str, timeframe: str):
    """Read finance_client's on-disk yfinance CSV cache directly, bypassing
    YahooClient.download()/CSVClient._get_ohlc_from_client()'s length=None bulk-retrieval path --
    that path is built around the CSV client's simulation-stepping API (self._step_index,
    "CSV data exhausted") and returns only 1 row for a fresh index=0 request instead of the
    full cached history. ohlc.collect() still has to be called first (see prepare_datasets) to
    make YahooClient.__init__'s incremental __get_rates() actually populate/refresh this cache
    file on disk -- that part works fine and is the only part actually needed."""
    import pandas as pd
    from finance_client import frames as Frame
    from data.collectors.ohlc import _FRAME_MAP

    frame = _FRAME_MAP.get(timeframe, Frame.D1)
    frame_str = Frame.to_str(frame)
    cache_path = Path.cwd() / "data_source" / "yfinance" / f"yfinance_{symbol}_{frame_str}.csv"
    df = pd.read_csv(cache_path, parse_dates=["Datetime"], index_col="Datetime")
    df.index.name = "datetime"
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    return df


# ---------------------------------------------------------------------------
# Phase 1: data prep -- no Celery worker required, calls collect() directly and registers the
# Dataset row the same way celery_worker.py's run_collection_job does for a non-incremental,
# non-DDM-streaming collector.
# ---------------------------------------------------------------------------

async def prepare_datasets() -> None:
    import database
    from data.models import Datasource, Dataset

    async with database.async_session_factory() as db:
        usdjpy_source = Datasource(
            name="USDJPY M1 (transfer experiment)",
            type="ohlc_download",
            config={"client": "yfinance", "symbol": USDJPY_SYMBOL, "timeframe": USDJPY_TIMEFRAME},
        )
        db.add(usdjpy_source)
        await db.flush()
        await db.refresh(usdjpy_source)

        await db.commit()
        usdjpy_source_id = usdjpy_source.id

    print(f"Created Datasource id={usdjpy_source_id} (USDJPY)")

    from data.collectors import ohlc

    print("Collecting USDJPY via yfinance (M1 intraday history is provider-limited -- expect "
          "only the last several weeks, not years)...")
    try:
        ohlc.collect(usdjpy_source_id, {
            "client": "yfinance", "symbol": USDJPY_SYMBOL, "timeframe": USDJPY_TIMEFRAME,
        })
    except RuntimeError:
        # "No data returned" can fire on collect()'s own (broken) length=None read; the on-disk
        # cache it refreshed as a side effect is what we actually read below.
        pass
    usdjpy_df = _read_cached_yfinance_ohlc(USDJPY_SYMBOL, USDJPY_TIMEFRAME)

    ddm_combined, ddm_from_ts, ddm_to_ts = _simulate_ddm_pretrain_data()

    async with database.async_session_factory() as db:
        import os
        store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
        usdjpy_artifact_rel = f"datasets/src_{usdjpy_source_id}/{USDJPY_SYMBOL.replace('/', '_')}_{USDJPY_TIMEFRAME}.parquet"
        (store / usdjpy_artifact_rel).parent.mkdir(parents=True, exist_ok=True)
        usdjpy_df.to_parquet(store / usdjpy_artifact_rel)

        usdjpy_ds = Dataset(
            datasource_id=usdjpy_source_id, name="USDJPY M1 (transfer experiment)",
            symbol=USDJPY_SYMBOL, timeframe=USDJPY_TIMEFRAME,
            from_ts=usdjpy_df.index[0].to_pydatetime(), to_ts=usdjpy_df.index[-1].to_pydatetime(),
            row_count=len(usdjpy_df), artifact_path=usdjpy_artifact_rel,
            status="ready",
        )
        db.add(usdjpy_ds)

        ddm_artifact_rel = "datasets/derived/ddm_v3_shock_transfer_pretrain.parquet"
        (store / ddm_artifact_rel).parent.mkdir(parents=True, exist_ok=True)
        ddm_combined.to_parquet(store / ddm_artifact_rel)

        ddm_ds = Dataset(
            datasource_id=None, name="DDM v3_shock (transfer experiment pretrain)",
            symbol="DDM-SYNTH", timeframe="M1",
            from_ts=ddm_from_ts, to_ts=ddm_to_ts,
            row_count=len(ddm_combined), artifact_path=ddm_artifact_rel,
            status="ready",
        )
        db.add(ddm_ds)
        await db.commit()
        await db.refresh(usdjpy_ds)
        await db.refresh(ddm_ds)

    print(f"USDJPY dataset id={usdjpy_ds.id} row_count={usdjpy_ds.row_count} "
          f"span={usdjpy_ds.from_ts} .. {usdjpy_ds.to_ts}")
    print(f"DDM dataset id={ddm_ds.id} row_count={ddm_ds.row_count} "
          f"span={ddm_ds.from_ts} .. {ddm_ds.to_ts}")
    print("\nPaste these into USDJPY_DATASET_ID / DDM_DATASET_ID at the top of this file.")


async def prepare_mixture_data() -> None:
    """Condition C's pretraining dataset: Sine + Delay + XOR + LFSR + DDM, MIXTURE_ROWS_PER_SOURCE
    candles each -- see _build_synthetic_mixture_data. No Celery worker or Datasource row needed
    (same direct-register pattern as the DDM-only dataset in prepare_datasets)."""
    import os

    import database
    from data.models import Dataset

    mixture_combined, mixture_from_ts, mixture_to_ts = _build_synthetic_mixture_data()

    async with database.async_session_factory() as db:
        store = Path(os.getenv("ARTIFACT_STORE_PATH", "../artifacts")).resolve()
        artifact_rel = "datasets/derived/synthetic_mixture_transfer_pretrain.parquet"
        (store / artifact_rel).parent.mkdir(parents=True, exist_ok=True)
        mixture_combined.to_parquet(store / artifact_rel)

        mixture_ds = Dataset(
            datasource_id=None, name="Synthetic mixture (transfer experiment pretrain, condition C)",
            symbol="MIXED-SYNTH", timeframe="M1",
            from_ts=mixture_from_ts, to_ts=mixture_to_ts,
            row_count=len(mixture_combined), artifact_path=artifact_rel,
            status="ready",
        )
        db.add(mixture_ds)
        await db.commit()
        await db.refresh(mixture_ds)

    print(f"Mixture dataset id={mixture_ds.id} row_count={mixture_ds.row_count} "
          f"span={mixture_ds.from_ts} .. {mixture_ds.to_ts}")
    print("\nPaste this into MIXTURE_DATASET_ID at the top of this file.")


# ---------------------------------------------------------------------------
# Phase 2: submit TrainingRuns
# ---------------------------------------------------------------------------

async def _ensure_model() -> int:
    if ML_MODEL_ID is not None:
        return ML_MODEL_ID
    import database
    from model.models import MLModel

    async with database.async_session_factory() as db:
        model = MLModel(name="decoder_only (transfer experiment)", architecture="decoder_only", config=MODEL_CONFIG)
        db.add(model)
        await db.commit()
        await db.refresh(model)
        print(f"Created MLModel id={model.id} -- paste into ML_MODEL_ID at the top of this file.")
        return model.id


async def _submit(model_id: int, dataset_id: int, hyperparams: dict, execution_target: str = "local") -> int:
    import database
    from model.repository import model_repo
    from celery_app import enqueue

    async with database.async_session_factory() as db:
        run = await model_repo.create_training_run(
            db, model_id=model_id, dataset_id=dataset_id,
            preprocessed_dataset_id=None, hyperparams=hyperparams, execution_target=execution_target,
        )
        await db.commit()
        run_id = run.id
    task_name = "colab_train_model" if execution_target == "colab" else "train_model"
    await enqueue(task_name, run_id)
    print(f"Submitted TrainingRun id={run_id} dataset_id={dataset_id} execution_target={execution_target} hyperparams={hyperparams}")
    return run_id


def _require_dataset_ids() -> None:
    if USDJPY_DATASET_ID is None or DDM_DATASET_ID is None:
        raise SystemExit(
            "USDJPY_DATASET_ID / DDM_DATASET_ID are not set -- run `prepare-data` first, "
            "then paste the printed dataset ids into this file."
        )


async def run_condition_a(seeds: list[int], max_steps: int, val_every_steps: int, execution_target: str = "local") -> list[int]:
    """Condition A: USDJPY from scratch."""
    model_id = await _ensure_model()
    run_ids = []
    for seed in seeds:
        hp = _finetune_hp(seed, warm_start_checkpoint=None)
        hp["max_steps"], hp["val_every_steps"] = max_steps, val_every_steps
        run_ids.append(await _submit(model_id, USDJPY_DATASET_ID, hp, execution_target))
    return run_ids


async def run_pretrain_then_finetune(
    pretrain_dataset_id: int,
    seeds: list[int], pretrain_seed: int,
    pretrain_max_steps: int, pretrain_val_every_steps: int,
    finetune_max_steps: int, finetune_val_every_steps: int,
    execution_target: str = "local",
) -> tuple[int, list[int]]:
    """Shared by condition B (pretrain_dataset_id=DDM_DATASET_ID) and condition C
    (pretrain_dataset_id=MIXTURE_DATASET_ID): pretrain (single seed) on pretrain_dataset_id ->
    USDJPY fine-tune (one run per seed in `seeds`).

    Waits for the pretrain run to reach a terminal status before submitting fine-tune runs, since
    they need its best.pt checkpoint path. Requires a Celery worker on the matching queue
    (`training` for execution_target="local", `colab` for "colab") to actually be running --
    this function only submits/polls, it does not run the training itself.
    """
    import database
    from sqlalchemy import select
    from model.models import TrainingRun

    model_id = await _ensure_model()
    pretrain_hp = _pretrain_hp(pretrain_seed, pretrain_max_steps, pretrain_val_every_steps)
    pretrain_run_id = await _submit(model_id, pretrain_dataset_id, pretrain_hp, execution_target)

    print(f"Waiting for pretrain run {pretrain_run_id} to complete "
          f"(requires a `{execution_target}`-queue Celery worker running)...")
    while True:
        async with database.async_session_factory() as db:
            run = (await db.execute(select(TrainingRun).where(TrainingRun.id == pretrain_run_id))).scalar_one()
            if run.status in ("completed", "error", "stopped"):
                break
        await asyncio.sleep(10)

    if run.status != "completed":
        raise RuntimeError(f"Pretrain run {pretrain_run_id} ended with status={run.status!r}, not proceeding to fine-tune")

    checkpoint_path = f"models/{model_id}/training_{pretrain_run_id}/best.pt"
    finetune_run_ids = []
    for seed in seeds:
        hp = _finetune_hp(seed, warm_start_checkpoint=checkpoint_path)
        hp["max_steps"], hp["val_every_steps"] = finetune_max_steps, finetune_val_every_steps
        finetune_run_ids.append(await _submit(model_id, USDJPY_DATASET_ID, hp, execution_target))
    return pretrain_run_id, finetune_run_ids


async def _run_smoke() -> None:
    seeds = [42]
    print("=== condition A (smoke) ===")
    await run_condition_a(seeds, max_steps=1000, val_every_steps=200)
    print("=== condition B (smoke) ===")
    await run_pretrain_then_finetune(
        DDM_DATASET_ID, seeds, pretrain_seed=42,
        pretrain_max_steps=1000, pretrain_val_every_steps=200,
        finetune_max_steps=1000, finetune_val_every_steps=200,
    )


async def _run_full(seeds: list[int]) -> None:
    # Both conditions local: the colab split (condition B via execution_target="colab", see
    # docs/colab-workflow.md) is implemented and verified via a local cell-extraction proxy test,
    # but blocked for a REAL run right now by an expired Google Drive OAuth token that needs an
    # interactive browser re-login (docs/colab-workflow.md "One-time setup" step 1) -- pass
    # execution_target="colab" to run_pretrain_then_finetune once that's done, to parallelize
    # condition A/B across the `training`/`colab` queues instead of running both serially here.
    print("=== condition A (full, local) ===")
    await run_condition_a(seeds, max_steps=FINETUNE_MAX_STEPS, val_every_steps=FINETUNE_VAL_EVERY_STEPS)
    print("=== condition B (full, local) ===")
    await run_pretrain_then_finetune(
        DDM_DATASET_ID, seeds, pretrain_seed=42,
        pretrain_max_steps=PRETRAIN_MAX_STEPS, pretrain_val_every_steps=PRETRAIN_VAL_EVERY_STEPS,
        finetune_max_steps=FINETUNE_MAX_STEPS, finetune_val_every_steps=FINETUNE_VAL_EVERY_STEPS,
    )


async def _run_condition_c(seeds: list[int]) -> None:
    """Priority (2), minimal 3-condition design (user-requested): condition C only -- A and B
    already exist from the regime_controlled sanity-check rerun (TrainingRuns 1439-1441 and
    1442-1445). Pretrains on the synthetic mixture (MIXTURE_DATASET_ID) instead of DDM alone,
    then fine-tunes on USDJPY with the exact same budget/seeds/architecture as A and B so the
    three conditions are directly comparable."""
    if MIXTURE_DATASET_ID is None:
        raise SystemExit(
            "MIXTURE_DATASET_ID is not set -- run `prepare-mixture-data` first, "
            "then paste the printed dataset id into this file."
        )
    print("=== condition C (mixture pretrain -> USDJPY fine-tune) ===")
    await run_pretrain_then_finetune(
        MIXTURE_DATASET_ID, seeds, pretrain_seed=42,
        pretrain_max_steps=PRETRAIN_MAX_STEPS, pretrain_val_every_steps=PRETRAIN_VAL_EVERY_STEPS,
        finetune_max_steps=FINETUNE_MAX_STEPS, finetune_val_every_steps=FINETUNE_VAL_EVERY_STEPS,
    )


async def _run_colab_smoke() -> None:
    """Minimal REAL-Colab check (not the extracted-cells local proxy already validated) -- one
    small DDM pretrain run via execution_target="colab", to confirm the actual colab-cli/Drive
    round trip works end-to-end (session creation, dataset snapshot export+download, the
    generated notebook's own execution, checkpoint retrieval) before committing to the full run's
    4-job colab sequence. Requires a `colab`-queue Celery worker running."""
    model_id = await _ensure_model()
    hp = _pretrain_hp(seed=42, max_steps=300, val_every_steps=100)
    run_id = await _submit(model_id, DDM_DATASET_ID, hp, execution_target="colab")
    print(f"Submitted colab smoke run {run_id} -- poll its status via the API or DB; "
          f"see docs/colab-workflow.md Step 3 for what to expect.")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if mode == "prepare-data":
        asyncio.run(prepare_datasets())
        return
    if mode == "prepare-mixture-data":
        asyncio.run(prepare_mixture_data())
        return

    _require_dataset_ids()

    if mode == "smoke":
        # Single asyncio.run() for the whole run, not one per condition -- database.
        # async_session_factory's pooled connections bind to whichever event loop created them,
        # so a separate asyncio.run() per condition leaves condition B trying to reuse a
        # connection whose loop already closed (same pitfall submit_regime_training.py's
        # submit_many() docstring already documents).
        asyncio.run(_run_smoke())
    elif mode == "colab-smoke":
        asyncio.run(_run_colab_smoke())
    elif mode == "full":
        seeds = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else SEEDS_FULL
        asyncio.run(_run_full(seeds))
    elif mode == "condition-c":
        seeds = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else SEEDS_FULL
        asyncio.run(_run_condition_c(seeds))
    else:
        raise SystemExit(
            f"unknown mode {mode!r}, expected 'prepare-data', 'prepare-mixture-data', "
            f"'smoke', 'colab-smoke', 'full', or 'condition-c'"
        )


if __name__ == "__main__":
    main()
