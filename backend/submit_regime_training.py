"""Throwaway: submit TrainingRuns against the ddm_regime_delta_regression dataset (see
generate_regime_datasets.py), reusing the same create-run + enqueue path the API router uses
(model_repo.create_training_run + celery_app.enqueue("train_model", run.id)) so this goes through
the real training pipeline without needing the FastAPI server up.

Usage:
    python submit_regime_training.py smoke            # 1 run, few epochs, sanity-check only
    python submit_regime_training.py full             # 3-seed batch, full epoch budget
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATASET_ID = 48
ML_MODEL_ID = 99
N_TOTAL_ROWS = 100_000

BASE_HP = {
    "obs_len": 60,
    "pred_len": 1,
    "feature_cols": ["close"],
    "tgt_feature_cols": ["delta"],
    "token_level": "quantize_diff",
    # "normalize" governs the TARGET column here (not the src -- token_level="quantize_diff"
    # tokenizes src from raw prices directly, bypassing normalize/src_normalize entirely). The
    # default "returns" applies np.log() to the target, which is fatal for delta (always <= 0):
    # log of a negative number is NaN. delta is a raw regression target, not a price series, so
    # it must pass through unnormalized.
    "normalize": "none",
    # NOT 0.2 (the "clean" 20/100 regime fraction): require_contiguous drops a slightly
    # different number of boundary windows on the train side vs. the held-out side, so 0.2 of
    # the post-filtering window count lands 1 window short of the true train/held-out regime
    # seam -- empirically verified to leak exactly 1 last-train-regime window into val. 0.1999
    # was found by direct verification (checking every val window's target against both regimes'
    # delta sets) to land exactly on the seam with zero leakage in either direction.
    "val_split": 0.1999,
    "split_mode": "chronological",
    "require_contiguous": True,
    "max_rows": N_TOTAL_ROWS,
    "batch_size": 64,
}

SEEDS_FULL = [42, 43, 44]


async def submit(hyperparams: dict, model_id: int = ML_MODEL_ID) -> int:
    import database
    from model.repository import model_repo
    from celery_app import enqueue

    async with database.async_session_factory() as db:
        run = await model_repo.create_training_run(
            db, model_id=model_id, dataset_id=DATASET_ID,
            preprocessed_dataset_id=None, hyperparams=hyperparams, execution_target="local",
        )
        await db.commit()
        run_id = run.id
    await enqueue("train_model", run_id)
    print(f"Submitted TrainingRun id={run_id} hyperparams={hyperparams}")
    return run_id


async def submit_many(hp_list: list[dict]) -> None:
    # One event loop for the whole batch -- database.async_session_factory's pooled connections
    # are bound to whichever loop created them, so a separate asyncio.run() per submission (the
    # original approach here) leaves later iterations trying to reuse a connection whose loop
    # already closed ("Event loop is closed" / AttributeError deep in asyncpg).
    for hp in hp_list:
        await submit(hp)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    seeds = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else SEEDS_FULL
    if mode == "smoke":
        hp = {**BASE_HP, "seed": 42, "epochs": 3}
        asyncio.run(submit(hp))
    elif mode == "full":
        hp_list = [{**BASE_HP, "seed": seed, "epochs": 30, "early_stop_patience": 5} for seed in seeds]
        asyncio.run(submit_many(hp_list))
    else:
        raise SystemExit(f"unknown mode {mode!r}, expected 'smoke' or 'full'")


if __name__ == "__main__":
    main()
