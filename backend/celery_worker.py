"""
AlgoForge Celery worker — background job tasks.

Start per-queue workers (from backend/):
    celery -A celery_worker worker -Q collection      -c 3  --pool=prefork --loglevel=info
    celery -A celery_worker worker -Q characteristics -c 12 --pool=prefork --loglevel=info
    celery -A celery_worker worker -Q training        -c 2  --pool=prefork --loglevel=info
    celery -A celery_worker worker -Q backtest        -c 5  --pool=prefork --loglevel=info

Each task runs in a separate OS process (prefork pool). Heavy/blocking work in
one process cannot affect other queues. asyncio.run() gives each task its own
event loop; SQLAlchemy uses NullPool so connections are not shared across loops.
"""

from __future__ import annotations

# Path bootstrap — before any application imports.
# resolve() gives a canonical absolute path; unconditional insert avoids
# Windows case/format mismatches that break "not in" checks.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio
import logging
import os
from datetime import datetime, timezone

from celery_app import celery_app

logger = logging.getLogger("celery_worker")


# ---------------------------------------------------------------------------
# Per-task DB helper — NullPool avoids asyncio event-loop / pool conflicts
# ---------------------------------------------------------------------------

def _make_db():
    """Create a fresh async engine + session factory for the current event loop.

    NullPool is SQLAlchemy's recommended approach for multiprocessing contexts:
    connections are opened per-use and closed immediately, so nothing is shared
    between tasks or across asyncio.run() calls.
    """
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.pool import NullPool

    url = os.environ.get("DATABASE_URL", "postgresql+asyncpg://algoforge:algoforge@localhost:5432/algoforge")
    engine = create_async_engine(url, poolclass=NullPool)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return factory, engine


# ---------------------------------------------------------------------------
# Dedup lock release helper
# ---------------------------------------------------------------------------

def _release_lock(task_name: str, entity_id) -> None:
    """Remove the Redis dedup lock set by celery_app.enqueue()."""
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        r.delete(f"algoforge:enqueued:{task_name}:{entity_id}")
    except Exception:
        pass  # Non-critical: lock expires after 1 h anyway


# ---------------------------------------------------------------------------
# Task 1 — run_collection_job
# ---------------------------------------------------------------------------

@celery_app.task(name="celery_worker.run_collection_job", bind=False)
def run_collection_job(job_id: int) -> dict:
    # For endless DDM jobs the task never returns, so the Redis dedup lock
    # (TTL=1h) would expire and allow a second concurrent instance to start.
    # Renew it every 30 min in a background thread for the lifetime of this task.
    import threading

    _lock_key = f"algoforge:enqueued:run_collection_job:{job_id}"
    _stop_renewer = threading.Event()

    def _renew_lock():
        while not _stop_renewer.wait(timeout=1800):  # every 30 min
            try:
                import redis as _r
                r = _r.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
                r.expire(_lock_key, 3600)
            except Exception:
                pass

    renewer = threading.Thread(target=_renew_lock, daemon=True)
    renewer.start()
    try:
        return asyncio.run(_run_collection_job(job_id))
    finally:
        _stop_renewer.set()


async def _run_collection_job(job_id: int) -> dict:
    from sqlalchemy import select, update
    from data.models import CollectionJob, Datasource, Dataset

    factory, engine = _make_db()
    try:
        async with factory() as db:
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

            symbol = source.config.get("symbol") or source.config.get("timeframe")
            timeframe = source.config.get("timeframe")
            is_ddm = source.type == "ddm_simulation"
            pre_dataset_id = None
            incremental_context = None

            await db.execute(update(CollectionJob).where(CollectionJob.id == job_id).values(status="running", last_error=None))

            if source.type == "ohlc_download":
                existing_ds = (await db.execute(
                    select(Dataset)
                    .where(Dataset.datasource_id == source.id)
                    .where(Dataset.status == "ready")
                    .where(Dataset.to_ts.isnot(None))
                    .where(Dataset.artifact_path.isnot(None))
                    .order_by(Dataset.id.desc())
                    .limit(1)
                )).scalar_one_or_none()
                if existing_ds is not None:
                    incremental_context = {
                        "dataset_id": existing_ds.id,
                        "artifact_path": existing_ds.artifact_path,
                        "to_ts": existing_ds.to_ts,
                        "from_ts": existing_ds.from_ts,
                    }
                    await db.execute(
                        update(Dataset).where(Dataset.id == existing_ds.id).values(status="running")
                    )

            if is_ddm:
                artifact_rel = f"datasets/src_{source.id}/ddm_ticks"
                existing = (await db.execute(
                    select(Dataset)
                    .where(Dataset.datasource_id == source.id)
                    .where(Dataset.artifact_path == artifact_rel)
                    .where(Dataset.status.in_(["running", "ready"]))
                    .order_by(Dataset.id.desc())
                    .limit(1)
                )).scalar_one_or_none()
                if existing is not None:
                    await db.execute(
                        update(Dataset).where(Dataset.id == existing.id).values(
                            name=f"{source.name} (running…)", status="running", row_count=0
                        )
                    )
                    pre_dataset_id = existing.id
                else:
                    pre_ds = Dataset(
                        datasource_id=source.id,
                        name=f"{source.name} (running…)",
                        symbol=symbol,
                        timeframe=timeframe,
                        row_count=0,
                        artifact_path=artifact_rel,
                        status="running",
                    )
                    db.add(pre_ds)
                    await db.flush()
                    await db.refresh(pre_ds)
                    pre_dataset_id = pre_ds.id

            await db.commit()

        collect_result = None
        try:
            import functools
            logger.info(f"[job {job_id}] starting collector: type={source.type} incremental={incremental_context is not None} config={source.config}")
            loop = asyncio.get_event_loop()
            collect_result = await loop.run_in_executor(
                None,
                functools.partial(_run_collector, source.type, source.id, source.config, incremental_context),
            )
            logger.info(f"[job {job_id}] collector finished: {collect_result}")
        except NotImplementedError as e:
            async with factory() as db:
                await db.execute(
                    update(CollectionJob).where(CollectionJob.id == job_id).values(
                        status="error", last_error=str(e), last_run_at=datetime.now(timezone.utc)
                    )
                )
                await db.commit()
            return {"error": str(e)}
        except BaseException as e:
            logger.exception(f"Collection job {job_id} failed")
            try:
                async with factory() as db:
                    await db.execute(
                        update(CollectionJob).where(CollectionJob.id == job_id).values(
                            status="error", last_error=str(e), last_run_at=datetime.now(timezone.utc)
                        )
                    )
                    if incremental_context is not None:
                        await db.execute(
                            update(Dataset).where(Dataset.id == incremental_context["dataset_id"]).values(
                                status="error"
                            )
                        )
                    elif pre_dataset_id is not None:
                        from data.collectors.ddm_simulator import read_meta
                        from datetime import datetime as dt
                        meta = read_meta(source.id)
                        def _parse_ts(val):
                            if val is None:
                                return None
                            if isinstance(val, str):
                                return dt.fromisoformat(val)
                            return val
                        await db.execute(
                            update(Dataset).where(Dataset.id == pre_dataset_id).values(
                                status="ready",
                                row_count=meta.get("row_count", 0),
                                to_ts=_parse_ts(meta.get("to_ts")),
                                from_ts=_parse_ts(meta.get("from_ts")),
                                name=f"{source.name} (partial)",
                            )
                        )
                    await db.commit()
            except Exception:
                logger.exception(f"Collection job {job_id}: secondary error updating DB after failure (ignored)")
            return {"error": str(e)}

        async with factory() as db:
            if incremental_context is not None:
                # Incremental: update the existing dataset row; preserve original from_ts
                await db.execute(
                    update(Dataset).where(Dataset.id == incremental_context["dataset_id"]).values(
                        name=f"{source.name} {incremental_context['from_ts'].date()} to {collect_result.to_ts.date()}",
                        to_ts=collect_result.to_ts,
                        row_count=collect_result.row_count,
                        status="ready",
                    )
                )
                dataset_id = incremental_context["dataset_id"]
            elif pre_dataset_id is not None:
                await db.execute(
                    update(Dataset).where(Dataset.id == pre_dataset_id).values(
                        name=f"{source.name} {collect_result.from_ts.date()} to {collect_result.to_ts.date()}",
                        from_ts=collect_result.from_ts,
                        to_ts=collect_result.to_ts,
                        row_count=collect_result.row_count,
                        status="ready",
                    )
                )
                dataset_id = pre_dataset_id
            else:
                dataset = Dataset(
                    datasource_id=source.id,
                    name=f"{source.name} {collect_result.from_ts.date()} to {collect_result.to_ts.date()}",
                    symbol=symbol,
                    timeframe=timeframe,
                    from_ts=collect_result.from_ts,
                    to_ts=collect_result.to_ts,
                    row_count=collect_result.row_count,
                    artifact_path=collect_result.artifact_path,
                    status="ready",
                )
                db.add(dataset)
                await db.flush()
                await db.refresh(dataset)
                dataset_id = dataset.id

            next_run_at = _compute_next_run(source.type, source.config, job.schedule_cron)
            await db.execute(
                update(CollectionJob).where(CollectionJob.id == job_id).values(
                    status="idle", last_run_at=datetime.now(timezone.utc), last_error=None,
                    next_run_at=next_run_at,
                )
            )
            await db.commit()

        logger.info(f"Collection job {job_id} completed: dataset {dataset_id}, {collect_result.row_count} rows")

        # Auto-trigger characteristics computation for all datasource types that
        # produce numeric time-series data. Skip economic_calendar — its schema
        # (indicator / value / unit) is not what the characteristics system expects.
        if source.type != "economic_calendar":
            compute_characteristics.apply_async(args=[dataset_id], queue="characteristics")
            logger.info(f"[job {job_id}] auto-enqueued compute_characteristics for dataset {dataset_id}")

        return {"dataset_id": dataset_id, "row_count": collect_result.row_count}
    finally:
        await engine.dispose()
        _release_lock("run_collection_job", job_id)


def _compute_next_run(datasource_type: str, config: dict, schedule_cron: str | None) -> "datetime | None":
    """Return the next UTC datetime this job should run, or None (one-off)."""
    from datetime import timedelta

    # web_report uses interval_days from the datasource config
    if datasource_type == "web_report":
        interval_days = config.get("interval_days")
        if interval_days is None:
            return None  # one-off download
        try:
            return datetime.now(timezone.utc) + timedelta(days=int(interval_days))
        except (TypeError, ValueError):
            return None

    # Other types use schedule_cron on the job (e.g. "0 9 * * *")
    if schedule_cron:
        try:
            from apscheduler.triggers.cron import CronTrigger
            trigger = CronTrigger.from_crontab(schedule_cron, timezone="UTC")
            return trigger.get_next_fire_time(None, datetime.now(timezone.utc))
        except Exception:
            logger.warning(f"_compute_next_run: invalid schedule_cron {schedule_cron!r}")

    return None


def _run_collector(datasource_type: str, datasource_id: int, config: dict, incremental: dict | None = None):
    """Dispatch to the correct collector synchronously."""
    if datasource_type == "ohlc_download":
        from data.collectors.ohlc import collect
        return collect(datasource_id, config, incremental=incremental)
    elif datasource_type == "ddm_simulation":
        from data.collectors.ddm_simulator import collect
        return collect(datasource_id, config)
    elif datasource_type == "web_report":
        from data.collectors.web_report import collect
        # Always force on manual runs; interval_days is enforced by the scheduler
        return collect(datasource_id, config, force=True)
    elif datasource_type == "economic_calendar":
        from data.collectors.economic_calendar import collect
        return collect(datasource_id, config)
    else:
        raise ValueError(f"Unknown datasource type: {datasource_type!r}")


# ---------------------------------------------------------------------------
# Task 2 — compute_characteristics
# ---------------------------------------------------------------------------

@celery_app.task(name="celery_worker.compute_characteristics", bind=False)
def compute_characteristics(dataset_id: int) -> dict:
    return asyncio.run(_compute_characteristics(dataset_id))


async def _compute_characteristics(dataset_id: int) -> dict:
    import time
    from sqlalchemy import select
    from data.models import Dataset, DataCharacteristics
    from data.characteristics import load_df_for_dataset, CHARACTERISTIC_REGISTRY

    logger.info(f"compute_characteristics started for dataset {dataset_id}")
    t0 = time.monotonic()

    factory, engine = _make_db()
    try:
        # ── 1. Resolve artifact path ───────────────────────────────────────────
        async with factory() as db:
            result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
            dataset = result.scalar_one_or_none()
            if dataset is None or dataset.artifact_path is None:
                logger.warning(f"compute_characteristics: dataset {dataset_id} not found or has no artifact")
                return {"error": "dataset_not_found_or_no_artifact"}
            artifact_path = dataset.artifact_path

        # ── 2. Load DataFrame once (may be slow for large datasets) ───────────
        try:
            df = await asyncio.to_thread(load_df_for_dataset, artifact_path)
        except Exception as e:
            logger.exception(f"Failed to load dataset {dataset_id}")
            return {"error": str(e)}

        # ── 3. Create an empty characteristics row so the UI can start polling ─
        async with factory() as db:
            char = DataCharacteristics(dataset_id=dataset_id, metrics={})
            db.add(char)
            await db.commit()
            await db.refresh(char)
            char_id = char.id

        # ── 4. Run each analysis and save to DB as it completes ────────────────
        for name, fn in CHARACTERISTIC_REGISTRY.items():
            t1 = time.monotonic()
            try:
                result = await asyncio.to_thread(fn, df)
            except Exception as e:
                logger.warning(f"Analysis '{name}' failed for dataset {dataset_id}: {e}")
                result = {"error": str(e)}

            async with factory() as db:
                char = await db.get(DataCharacteristics, char_id)
                char.metrics = {**char.metrics, name: result}
                await db.commit()

            logger.debug(f"  [{name}] done in {time.monotonic() - t1:.2f}s")

        elapsed = time.monotonic() - t0
        logger.info(f"compute_characteristics done for dataset {dataset_id} in {elapsed:.1f}s → id={char_id}")
        return {"characteristics_id": char_id}
    finally:
        await engine.dispose()
        _release_lock("compute_characteristics", dataset_id)


# ---------------------------------------------------------------------------
# Task 3 — tick_scheduler (Celery Beat, runs every 60 s)
# ---------------------------------------------------------------------------

@celery_app.task(name="celery_worker.tick_scheduler", bind=False)
def tick_scheduler() -> None:
    """Enqueue any collection jobs whose next_run_at is now due."""
    asyncio.run(_tick_scheduler())


async def _tick_scheduler() -> None:
    from sqlalchemy import select
    from data.models import CollectionJob

    now = datetime.now(timezone.utc)
    factory, engine = _make_db()
    try:
        async with factory() as db:
            result = await db.execute(
                select(CollectionJob).where(
                    CollectionJob.enabled == True,  # noqa: E712
                    CollectionJob.next_run_at.isnot(None),
                    CollectionJob.next_run_at <= now,
                    CollectionJob.status != "running",
                )
            )
            jobs = result.scalars().all()

        for job in jobs:
            try:
                _release_lock("run_collection_job", job.id)
                from celery_app import enqueue
                await enqueue("run_collection_job", job.id)
                logger.info(f"[tick_scheduler] enqueued job {job.id} (next_run_at={job.next_run_at})")
            except Exception as exc:
                logger.warning(f"[tick_scheduler] could not enqueue job {job.id}: {exc}")
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Task 4 — train_model
# ---------------------------------------------------------------------------

@celery_app.task(name="celery_worker.train_model", bind=False)
def train_model(training_run_id: int) -> dict:
    return asyncio.run(_train_model(training_run_id))


async def _train_model(training_run_id: int) -> dict:
    import torch
    from sqlalchemy import select, update
    from model.models import MLModel, TrainingRun, TrainingCheckpoint
    from model.architectures import build_model, TRAINING_DEFAULTS
    from model.trainers import get_trainer_fns, OHLCWindowDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))

    factory, engine = _make_db()
    try:
        async with factory() as db:
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
            async with factory() as db:
                await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                    status="error", ended_at=datetime.now(timezone.utc)
                ))
                await db.commit()
            return {"error": str(e)}

        train_fn, eval_fn = get_trainer_fns(architecture)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.get("lr", 0.001))
        criterion = None if architecture in ("timegan", "vae") else torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5) if criterion else None

        epochs = int(hp.get("epochs", 50))
        batch_size = int(hp.get("batch_size", 32))
        best_val_loss = float("inf")
        best_epoch = 0
        checkpoint_dir = store / "models" / str(model_rec.id) / f"training_{training_run_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_event_loop()
        for epoch in range(1, epochs + 1):
            # Check for stop request before each epoch
            async with factory() as db:
                stop_result = await db.execute(
                    __import__("sqlalchemy", fromlist=["select"]).select(
                        TrainingRun.stop_requested
                    ).where(TrainingRun.id == training_run_id)
                )
                if stop_result.scalar():
                    logger.info(f"Training run {training_run_id} stopped at epoch {epoch}")
                    break

            train_loss = await loop.run_in_executor(None, train_fn, model, dataset, optimizer, criterion, batch_size)
            val_loss = await loop.run_in_executor(None, eval_fn, model, dataset, criterion, batch_size)

            if scheduler:
                scheduler.step(val_loss)

            ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss}, ckpt_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss}, checkpoint_dir / "best.pt")

            eta_seconds = (epochs - epoch) * 5

            async with factory() as db:
                await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                    current_epoch=epoch, val_loss=val_loss, best_epoch=best_epoch, eta_seconds=eta_seconds
                ))
                db.add(TrainingCheckpoint(
                    training_run_id=training_run_id,
                    epoch=epoch,
                    metrics={"train_loss": train_loss, "val_loss": val_loss},
                    artifact_path=str(ckpt_path.relative_to(store)),
                ))
                await db.commit()

            logger.info(f"Training run {training_run_id} epoch {epoch}/{epochs}: train={train_loss:.6f} val={val_loss:.6f}")

        best_artifact = str((checkpoint_dir / "best.pt").relative_to(store))
        async with factory() as db:
            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                status="completed", ended_at=datetime.now(timezone.utc),
                best_epoch=best_epoch, val_loss=best_val_loss,
                artifact_path=best_artifact, eta_seconds=0,
            ))
            await db.execute(update(MLModel).where(MLModel.id == model_rec.id).values(status="trained"))
            await db.commit()

        logger.info(f"Training run {training_run_id} completed. Best epoch: {best_epoch}, val_loss: {best_val_loss:.6f}")
        return {"best_epoch": best_epoch, "val_loss": best_val_loss, "artifact_path": best_artifact}
    finally:
        await engine.dispose()
        _release_lock("train_model", training_run_id)


# ---------------------------------------------------------------------------
# Task 4 — validate_model
# ---------------------------------------------------------------------------

@celery_app.task(name="celery_worker.validate_model", bind=False)
def validate_model(validation_id: int) -> dict:
    """Validate a trained model. Takes a ModelValidation record ID."""
    return asyncio.run(_validate_model(validation_id))


async def _validate_model(validation_id: int) -> dict:
    from sqlalchemy import select
    from model.models import MLModel, TrainingRun, ModelValidation
    from model.validation import run_validation
    from data.models import Dataset

    factory, engine = _make_db()
    try:
        async with factory() as db:
            result = await db.execute(select(ModelValidation).where(ModelValidation.id == validation_id))
            val_rec = result.scalar_one_or_none()
            if val_rec is None:
                return {"error": "validation_not_found"}

            result = await db.execute(select(MLModel).where(MLModel.id == val_rec.model_id))
            model_rec = result.scalar_one_or_none()
            result2 = await db.execute(select(TrainingRun).where(TrainingRun.id == val_rec.training_run_id))
            run = result2.scalar_one_or_none()
            result3 = await db.execute(select(Dataset).where(Dataset.id == val_rec.dataset_id))
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
                logger.exception(f"Validation failed for validation_id {validation_id}")
                return {"error": str(e)}

            from sqlalchemy import update
            await db.execute(
                update(ModelValidation).where(ModelValidation.id == validation_id).values(metrics=metrics)
            )
            await db.commit()

        return {"validation_id": validation_id, "metrics": metrics}
    finally:
        await engine.dispose()
        _release_lock("validate_model", validation_id)


# ---------------------------------------------------------------------------
# Task 5 — execute_strategy_run
# ---------------------------------------------------------------------------

@celery_app.task(name="celery_worker.execute_strategy_run", bind=False)
def execute_strategy_run(run_id: int) -> dict:
    return asyncio.run(_execute_strategy_run(run_id))


async def _execute_strategy_run(run_id: int) -> dict:
    from strategy.executor import execute_strategy_run as _execute
    session_factory, engine = _make_db()
    try:
        return await _execute(run_id, session_factory=session_factory)
    finally:
        await engine.dispose()
        _release_lock("execute_strategy_run", run_id)
