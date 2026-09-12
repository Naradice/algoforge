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
USDJPY_DATASET_ID: int | None = 51
DDM_DATASET_ID: int | None = 52
# Filled in by `prepare-data`/first submit -- the shared decoder_only MLModel both conditions'
# runs are created under (same architecture config = same warm-started weight shapes).
ML_MODEL_ID: int | None = None

VOL_PERIOD = 20  # bars; also fixes the tgt_feature_cols column name "vol_{VOL_PERIOD}"

USDJPY_SYMBOL = "USDJPY=X"  # yfinance forex ticker format (plain "USDJPY" resolves to no data)
# "M1" matches the scope already fixed for this follow-up in
# docs/research-seed-five-axes-of-scaling.md's seed-q1
USDJPY_TIMEFRAME = "M1"

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
    "split_mode": "chronological",
    "require_contiguous": True,
    "val_split": 0.2,
    "batch_size": 64,
    "disable_lr_scheduler": True,
    # NOTE: vol_{PERIOD}'s first (PERIOD-1) rows are NaN (rolling std warm-up) -- preprocessing.py
    # doesn't drop them and OHLCWindowDataset's require_contiguous gap-mask is timestamp-gap-based,
    # not NaN-based, so a handful of early windows can carry a NaN target. With PERIOD=20 against
    # a dataset of hundreds of thousands of rows this is noise-level (occasionally pollutes one
    # train_loss checkpoint average via np.mean, never val_loss since val is the chronological
    # tail) -- verify this doesn't show up as a NaN val_loss during the smoke run before ignoring it.
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


def _simulate_ddm_pretrain_data():
    """DDM_N_RUNS independent, freshly-seeded DDMv3 (v3_shock params) runs of
    DDM_CANDLES_PER_RUN candles each, concatenated with a timestamp gap between every pair --
    see DDM_PRETRAIN_ROWS's comment for why not one long run. Returns (combined_df, from_ts,
    to_ts). Mirrors generate_regime_datasets.py's simulate/trades_to_ohlc/build_combined, which
    established this exact pattern for the same reason (avoiding DDMv3's long-horizon WMA
    divergence)."""
    import numpy as np
    import pandas as pd
    from data.collectors.ddm_simulator import DDMv3

    n_trades_per_run = DDM_CANDLES_PER_RUN * DDM_TRADES_PER_CANDLE
    cursor_ts = pd.Timestamp("2000-01-03 00:00:00", tz="UTC")
    blocks = []
    for run_idx in range(DDM_N_RUNS):
        seed = 1000 + run_idx
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
        }).iloc[:DDM_CANDLES_PER_RUN]

        idx = cursor_ts + pd.to_timedelta(np.arange(len(ohlc)) * 60, unit="s")
        ohlc.index = idx
        ohlc.index.name = "datetime"
        blocks.append(ohlc)
        cursor_ts = idx[-1] + pd.Timedelta(seconds=DDM_GAP_SECONDS)
        print(f"  DDM pretrain run {run_idx + 1}/{DDM_N_RUNS}: {len(ohlc)} candles")

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


async def run_condition_b(
    seeds: list[int], pretrain_seed: int,
    pretrain_max_steps: int, pretrain_val_every_steps: int,
    finetune_max_steps: int, finetune_val_every_steps: int,
    execution_target: str = "local",
) -> tuple[int, list[int]]:
    """Condition B: DDM pretrain (single seed) -> USDJPY fine-tune (one run per seed in `seeds`).

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
    pretrain_run_id = await _submit(model_id, DDM_DATASET_ID, pretrain_hp, execution_target)

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
    await run_condition_b(
        seeds, pretrain_seed=42,
        pretrain_max_steps=1000, pretrain_val_every_steps=200,
        finetune_max_steps=1000, finetune_val_every_steps=200,
    )


async def _run_full(seeds: list[int]) -> None:
    # Both conditions local: the colab split (condition B via execution_target="colab", see
    # docs/colab-workflow.md) is implemented and verified via a local cell-extraction proxy test,
    # but blocked for a REAL run right now by an expired Google Drive OAuth token that needs an
    # interactive browser re-login (docs/colab-workflow.md "One-time setup" step 1) -- pass
    # execution_target="colab" to run_condition_b once that's done, to parallelize condition A/B
    # across the `training`/`colab` queues instead of running both serially here.
    print("=== condition A (full, local) ===")
    await run_condition_a(seeds, max_steps=FINETUNE_MAX_STEPS, val_every_steps=FINETUNE_VAL_EVERY_STEPS)
    print("=== condition B (full, local) ===")
    await run_condition_b(
        seeds, pretrain_seed=42,
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
    else:
        raise SystemExit(f"unknown mode {mode!r}, expected 'prepare-data', 'smoke', 'colab-smoke', or 'full'")


if __name__ == "__main__":
    main()
