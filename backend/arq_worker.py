"""
AlgoForge arq worker — background job functions.

Start with:
    cd finance/algoforge/backend
    python -m arq arq_worker.WorkerSettings

All heavy collection and training jobs run here instead of inside the API process.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from arq.connections import RedisSettings

logger = logging.getLogger("arq_worker")

# ---------------------------------------------------------------------------
# Data collection jobs
# ---------------------------------------------------------------------------


async def run_collection_job(ctx, job_id: int) -> dict:
    """
    Run a collection job:
    1. Load CollectionJob + Datasource config from DB
    2. Call the appropriate collector
    3. Create/update Dataset record with artifact path and row count
    4. Update CollectionJob.last_run_at and status
    """
    from database import async_session_factory
    from data.models import CollectionJob, Datasource, Dataset
    from sqlalchemy import select, update

    async with async_session_factory() as db:
        result = await db.execute(select(CollectionJob).where(CollectionJob.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            logger.error(f"CollectionJob {job_id} not found")
            return {"error": "job_not_found"}

        result = await db.execute(select(Datasource).where(Datasource.id == job.datasource_id))
        source = result.scalar_one_or_none()
        if source is None:
            logger.error(f"Datasource {job.datasource_id} not found for job {job_id}")
            return {"error": "datasource_not_found"}

        # Mark as running
        await db.execute(update(CollectionJob).where(CollectionJob.id == job_id).values(status="running"))
        await db.commit()

    try:
        collect_result = _run_collector(source.type, source.id, source.config)
    except NotImplementedError as e:
        async with async_session_factory() as db:
            await db.execute(
                update(CollectionJob).where(CollectionJob.id == job_id).values(
                    status="error", last_error=str(e), last_run_at=datetime.now(timezone.utc)
                )
            )
            await db.commit()
        return {"error": str(e)}
    except Exception as e:
        logger.exception(f"Collection job {job_id} failed")
        async with async_session_factory() as db:
            await db.execute(
                update(CollectionJob).where(CollectionJob.id == job_id).values(
                    status="error", last_error=str(e), last_run_at=datetime.now(timezone.utc)
                )
            )
            await db.commit()
        return {"error": str(e)}

    # Persist Dataset record
    async with async_session_factory() as db:
        symbol = source.config.get("symbol") or source.config.get("timeframe")
        timeframe = source.config.get("timeframe")
        dataset = Dataset(
            datasource_id=source.id,
            name=f"{source.name} – {collect_result.from_ts.date()} to {collect_result.to_ts.date()}",
            symbol=symbol,
            timeframe=timeframe,
            from_ts=collect_result.from_ts,
            to_ts=collect_result.to_ts,
            row_count=collect_result.row_count,
            artifact_path=collect_result.artifact_path,
            status="ready",
        )
        db.add(dataset)
        await db.execute(
            update(CollectionJob).where(CollectionJob.id == job_id).values(
                status="idle", last_run_at=datetime.now(timezone.utc), last_error=None
            )
        )
        await db.commit()
        await db.refresh(dataset)
        dataset_id = dataset.id

    logger.info(f"Collection job {job_id} completed: dataset {dataset_id}, {collect_result.row_count} rows")
    return {"dataset_id": dataset_id, "row_count": collect_result.row_count}


def _run_collector(datasource_type: str, datasource_id: int, config: dict):
    """Dispatch to the correct collector synchronously."""
    if datasource_type == "ohlc_download":
        from data.collectors.ohlc import collect
    elif datasource_type == "ddm_simulation":
        from data.collectors.ddm_simulator import collect
    elif datasource_type == "web_report":
        from data.collectors.web_report import collect
    else:
        raise ValueError(f"Unknown datasource type: {datasource_type!r}")
    return collect(datasource_id, config)


# ---------------------------------------------------------------------------
# Characteristic analysis jobs
# ---------------------------------------------------------------------------


async def compute_characteristics(ctx, dataset_id: int) -> dict:
    """
    Run characteristic analysis on a dataset and persist the result.
    """
    from database import async_session_factory
    from data.models import Dataset, DataCharacteristics
    from data.characteristics import compute_for_dataset
    from sqlalchemy import select

    async with async_session_factory() as db:
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        if dataset is None or dataset.artifact_path is None:
            return {"error": "dataset_not_found_or_no_artifact"}

        try:
            metrics = compute_for_dataset(dataset.artifact_path)
        except Exception as e:
            logger.exception(f"Characteristic analysis failed for dataset {dataset_id}")
            return {"error": str(e)}

        char = DataCharacteristics(dataset_id=dataset_id, metrics=metrics)
        db.add(char)
        await db.commit()
        await db.refresh(char)

    logger.info(f"Characteristics computed for dataset {dataset_id}")
    return {"characteristics_id": char.id}


# ---------------------------------------------------------------------------
# Model training jobs
# ---------------------------------------------------------------------------


async def train_model(ctx, training_run_id: int) -> dict:
    """
    Train an ML model:
    1. Load TrainingRun + MLModel config from DB
    2. Build model + dataset
    3. Run epoch loop: update current_epoch + val_loss in DB after each epoch
    4. Save best checkpoint; on completion update TrainingRun + MLModel status
    """
    import asyncio
    import os
    import torch
    from pathlib import Path
    from database import async_session_factory
    from model.models import MLModel, TrainingRun, TrainingCheckpoint
    from model.architectures import build_model, TRAINING_DEFAULTS
    from model.trainers import get_trainer_fns, OHLCWindowDataset
    from sqlalchemy import select, update

    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))

    # ── Load config from DB ───────────────────────────────────────────────────
    async with async_session_factory() as db:
        result = await db.execute(select(TrainingRun).where(TrainingRun.id == training_run_id))
        run = result.scalar_one_or_none()
        if run is None:
            return {"error": "training_run_not_found"}
        result = await db.execute(select(MLModel).where(MLModel.id == run.model_id))
        model_rec = result.scalar_one_or_none()
        if model_rec is None:
            return {"error": "model_not_found"}

        architecture = model_rec.architecture
        model_config = model_rec.config
        dataset_id = run.dataset_id
        hp = {**TRAINING_DEFAULTS.get(architecture, {}), **run.hyperparams}

        # Get dataset artifact path
        from data.models import Dataset
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        ds_rec = result.scalar_one_or_none()
        if ds_rec is None or ds_rec.artifact_path is None:
            return {"error": "dataset_not_found_or_no_artifact"}
        dataset_artifact = ds_rec.artifact_path

        await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
            status="running", started_at=datetime.now(timezone.utc)
        ))
        await db.execute(update(MLModel).where(MLModel.id == model_rec.id).values(status="training"))
        await db.commit()

    # ── Build model + dataset (sync, outside DB session) ─────────────────────
    try:
        model = build_model(architecture, model_config, device=device)
        dataset = OHLCWindowDataset(
            dataset_artifact,
            obs_len=hp.get("obs_len", 60),
            pred_len=hp.get("pred_len", 10),
            feature_cols=hp.get("feature_cols", ["close"]),
            normalize=hp.get("normalize", "returns"),
            val_split=hp.get("val_split", 0.2),
            device=device,
        )
    except Exception as e:
        async with async_session_factory() as db:
            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                status="error", ended_at=datetime.now(timezone.utc)
            ))
            await db.commit()
        return {"error": str(e)}

    train_fn, eval_fn = get_trainer_fns(architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.get("lr", 0.001))
    criterion = torch.nn.MSELoss() if architecture != "timegan" else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5) if criterion else None

    epochs = int(hp.get("epochs", 50))
    batch_size = int(hp.get("batch_size", 32))
    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_dir = store / "models" / str(model_rec.id) / f"training_{training_run_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        loop = asyncio.get_event_loop()

        train_loss = await loop.run_in_executor(None, train_fn, model, dataset, optimizer, criterion, batch_size)
        val_loss = await loop.run_in_executor(None, eval_fn, model, dataset, criterion, batch_size)

        if scheduler:
            scheduler.step(val_loss)

        # Save checkpoint every epoch
        ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss}, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss}, checkpoint_dir / "best.pt")

        # Estimate ETA
        eta_per_epoch_s = 5  # rough estimate; could track actual time
        eta_seconds = (epochs - epoch) * eta_per_epoch_s

        async with async_session_factory() as db:
            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                current_epoch=epoch, val_loss=val_loss, best_epoch=best_epoch, eta_seconds=eta_seconds
            ))
            checkpoint_record = TrainingCheckpoint(
                training_run_id=training_run_id,
                epoch=epoch,
                metrics={"train_loss": train_loss, "val_loss": val_loss},
                artifact_path=str(ckpt_path.relative_to(store)),
            )
            db.add(checkpoint_record)
            await db.commit()

        logger.info(f"Training run {training_run_id} epoch {epoch}/{epochs}: train={train_loss:.6f} val={val_loss:.6f}")

    # ── Finalise ──────────────────────────────────────────────────────────────
    best_artifact = str((checkpoint_dir / "best.pt").relative_to(store))
    async with async_session_factory() as db:
        await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
            status="completed", ended_at=datetime.now(timezone.utc),
            best_epoch=best_epoch, val_loss=best_val_loss,
            artifact_path=best_artifact, eta_seconds=0,
        ))
        await db.execute(update(MLModel).where(MLModel.id == model_rec.id).values(status="trained"))
        await db.commit()

    logger.info(f"Training run {training_run_id} completed. Best epoch: {best_epoch}, val_loss: {best_val_loss:.6f}")
    return {"best_epoch": best_epoch, "val_loss": best_val_loss, "artifact_path": best_artifact}


async def validate_model(ctx, model_id: int, training_run_id: int, dataset_id: int) -> dict:
    """Compute validation metrics for a trained model checkpoint."""
    import torch
    from database import async_session_factory
    from model.models import MLModel, TrainingRun, ModelValidation
    from model.validation import run_validation
    from data.models import Dataset
    from sqlalchemy import select

    async with async_session_factory() as db:
        result = await db.execute(select(MLModel).where(MLModel.id == model_id))
        model_rec = result.scalar_one_or_none()
        result2 = await db.execute(select(TrainingRun).where(TrainingRun.id == training_run_id))
        run = result2.scalar_one_or_none()
        result3 = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        ds_rec = result3.scalar_one_or_none()

        if not model_rec or not run or not ds_rec:
            return {"error": "record_not_found"}

        try:
            metrics = run_validation(
                artifact_path=run.artifact_path,
                architecture=model_rec.architecture,
                model_config=model_rec.config,
                hyperparams=run.hyperparams,
                dataset_artifact_path=ds_rec.artifact_path,
            )
        except Exception as e:
            logger.exception(f"Validation failed for model {model_id}")
            return {"error": str(e)}

        val_record = ModelValidation(
            model_id=model_id, training_run_id=training_run_id,
            dataset_id=dataset_id, metrics=metrics,
        )
        db.add(val_record)
        await db.commit()
        await db.refresh(val_record)

    return {"validation_id": val_record.id, "metrics": metrics}


# ---------------------------------------------------------------------------
# Strategy execution jobs
# ---------------------------------------------------------------------------


async def execute_strategy_run(ctx, run_id: int) -> dict:
    """Run a strategy backtest (or paper/live in Phase 5)."""
    from strategy.executor import execute_strategy_run as _execute
    return await _execute(run_id)


# ---------------------------------------------------------------------------
# Worker settings
# ---------------------------------------------------------------------------


class WorkerSettings:
    functions = [run_collection_job, compute_characteristics, train_model, validate_model, execute_strategy_run]
    redis_settings = RedisSettings.from_dsn(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    max_jobs = 4
    job_timeout = 3600 * 6  # 6 hours (long training runs)
