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
USDJPY_DATASET_ID: int | None = None
DDM_DATASET_ID: int | None = None
# Filled in by `prepare-data`/first submit -- the shared decoder_only MLModel both conditions'
# runs are created under (same architecture config = same warm-started weight shapes).
ML_MODEL_ID: int | None = None

VOL_PERIOD = 20  # bars; also fixes the tgt_feature_cols column name "vol_{VOL_PERIOD}"

USDJPY_SYMBOL = "USDJPY=X"  # yfinance forex ticker format (plain "USDJPY" resolves to no data)
USDJPY_TIMEFRAME = "M1"     # matches the scope already fixed for this follow-up in
                             # docs/research-seed-five-axes-of-scaling.md's seed-q1

DDM_PRETRAIN_ROWS = 300_000  # pretraining budget is explicitly NOT required to match condition
                              # A's data volume -- only the fine-tune side must match (user's own
                              # Phase 2 spec) -- picked independently, generous since it's "free".

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

        ddm_source = Datasource(
            name="DDM v3_shock (transfer experiment pretrain)",
            type="ddm_simulation",
            config={"model": "v3_shock", "num_agent": 300, "length": DDM_PRETRAIN_ROWS,
                    "timeframe": "M1", "seed": 42},
        )
        db.add(ddm_source)
        await db.flush()
        await db.refresh(ddm_source)
        await db.commit()
        usdjpy_source_id, ddm_source_id = usdjpy_source.id, ddm_source.id

    print(f"Created Datasource id={usdjpy_source_id} (USDJPY) and id={ddm_source_id} (DDM)")

    from data.collectors import ohlc, ddm_simulator

    print("Collecting USDJPY via yfinance (M1 intraday history is provider-limited -- expect "
          "only the last several weeks, not years)...")
    usdjpy_result = ohlc.collect(usdjpy_source_id, {
        "client": "yfinance", "symbol": USDJPY_SYMBOL, "timeframe": USDJPY_TIMEFRAME,
    })

    print(f"Simulating DDM v3_shock ({DDM_PRETRAIN_ROWS} candles)...")
    ddm_result = ddm_simulator.collect(ddm_source_id, {
        "model": "v3_shock", "num_agent": 300, "length": DDM_PRETRAIN_ROWS,
        "timeframe": "M1", "seed": 42,
    })

    async with database.async_session_factory() as db:
        usdjpy_ds = Dataset(
            datasource_id=usdjpy_source_id, name="USDJPY M1 (transfer experiment)",
            symbol=USDJPY_SYMBOL, timeframe=USDJPY_TIMEFRAME,
            from_ts=usdjpy_result.from_ts, to_ts=usdjpy_result.to_ts,
            row_count=usdjpy_result.row_count, artifact_path=usdjpy_result.artifact_path,
            status="ready",
        )
        ddm_ds = Dataset(
            datasource_id=ddm_source_id, name="DDM v3_shock (transfer experiment pretrain)",
            symbol="DDM-SYNTH", timeframe="M1",
            from_ts=ddm_result.from_ts, to_ts=ddm_result.to_ts,
            row_count=ddm_result.row_count, artifact_path=ddm_result.artifact_path,
            status="ready",
        )
        db.add_all([usdjpy_ds, ddm_ds])
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


async def _submit(model_id: int, dataset_id: int, hyperparams: dict) -> int:
    import database
    from model.repository import model_repo
    from celery_app import enqueue

    async with database.async_session_factory() as db:
        run = await model_repo.create_training_run(
            db, model_id=model_id, dataset_id=dataset_id,
            preprocessed_dataset_id=None, hyperparams=hyperparams, execution_target="local",
        )
        await db.commit()
        run_id = run.id
    await enqueue("train_model", run_id)
    print(f"Submitted TrainingRun id={run_id} dataset_id={dataset_id} hyperparams={hyperparams}")
    return run_id


def _require_dataset_ids() -> None:
    if USDJPY_DATASET_ID is None or DDM_DATASET_ID is None:
        raise SystemExit(
            "USDJPY_DATASET_ID / DDM_DATASET_ID are not set -- run `prepare-data` first, "
            "then paste the printed dataset ids into this file."
        )


async def run_condition_a(seeds: list[int], max_steps: int, val_every_steps: int) -> list[int]:
    """Condition A: USDJPY from scratch."""
    model_id = await _ensure_model()
    run_ids = []
    for seed in seeds:
        hp = _finetune_hp(seed, warm_start_checkpoint=None)
        hp["max_steps"], hp["val_every_steps"] = max_steps, val_every_steps
        run_ids.append(await _submit(model_id, USDJPY_DATASET_ID, hp))
    return run_ids


async def run_condition_b(
    seeds: list[int], pretrain_seed: int,
    pretrain_max_steps: int, pretrain_val_every_steps: int,
    finetune_max_steps: int, finetune_val_every_steps: int,
) -> tuple[int, list[int]]:
    """Condition B: DDM pretrain (single seed) -> USDJPY fine-tune (one run per seed in `seeds`).

    Waits for the pretrain run to reach a terminal status before submitting fine-tune runs, since
    they need its best.pt checkpoint path. Requires a `training`-queue Celery worker to actually
    be running -- this function only submits/polls, it does not run the training itself.
    """
    import database
    from sqlalchemy import select
    from model.models import TrainingRun

    model_id = await _ensure_model()
    pretrain_hp = _pretrain_hp(pretrain_seed, pretrain_max_steps, pretrain_val_every_steps)
    pretrain_run_id = await _submit(model_id, DDM_DATASET_ID, pretrain_hp)

    print(f"Waiting for pretrain run {pretrain_run_id} to complete "
          f"(requires a `training`-queue Celery worker running)...")
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
        finetune_run_ids.append(await _submit(model_id, USDJPY_DATASET_ID, hp))
    return pretrain_run_id, finetune_run_ids


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if mode == "prepare-data":
        asyncio.run(prepare_datasets())
        return

    _require_dataset_ids()

    if mode == "smoke":
        seeds = [42]
        print("=== condition A (smoke) ===")
        asyncio.run(run_condition_a(seeds, max_steps=1000, val_every_steps=200))
        print("=== condition B (smoke) ===")
        asyncio.run(run_condition_b(
            seeds, pretrain_seed=42,
            pretrain_max_steps=1000, pretrain_val_every_steps=200,
            finetune_max_steps=1000, finetune_val_every_steps=200,
        ))
    elif mode == "full":
        seeds = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else SEEDS_FULL
        print("=== condition A (full) ===")
        asyncio.run(run_condition_a(seeds, max_steps=FINETUNE_MAX_STEPS, val_every_steps=FINETUNE_VAL_EVERY_STEPS))
        print("=== condition B (full) ===")
        asyncio.run(run_condition_b(
            seeds, pretrain_seed=42,
            pretrain_max_steps=PRETRAIN_MAX_STEPS, pretrain_val_every_steps=PRETRAIN_VAL_EVERY_STEPS,
            finetune_max_steps=FINETUNE_MAX_STEPS, finetune_val_every_steps=FINETUNE_VAL_EVERY_STEPS,
        ))
    else:
        raise SystemExit(f"unknown mode {mode!r}, expected 'prepare-data', 'smoke', or 'full'")


if __name__ == "__main__":
    main()
