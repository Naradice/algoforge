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
import math
import os
from datetime import datetime, timezone

from celery_app import celery_app
from webhooks.dispatcher import dispatch

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
                await dispatch(db, "collection.error", {
                    "collection_job_id": job_id, "datasource_id": source.id, "error": str(e),
                })
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
                    await dispatch(db, "collection.error", {
                        "collection_job_id": job_id, "datasource_id": source.id, "error": str(e),
                    })
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
            if source.type == "web_report":
                # web_report: one persistent dataset per datasource, updated on every run.
                # row_count = total files on disk; from_ts = first ever run; to_ts = this run.
                existing = (await db.execute(
                    select(Dataset)
                    .where(Dataset.datasource_id == source.id)
                    .order_by(Dataset.id)
                    .limit(1)
                )).scalar_one_or_none()
                if existing is not None:
                    await db.execute(
                        update(Dataset).where(Dataset.id == existing.id).values(
                            to_ts=collect_result.to_ts,
                            row_count=collect_result.row_count,
                            status="ready",
                        )
                    )
                    dataset_id = existing.id
                else:
                    dataset = Dataset(
                        datasource_id=source.id,
                        name=source.name,
                        symbol=None,
                        timeframe=None,
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
            elif incremental_context is not None:
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
            await dispatch(db, "collection.completed", {
                "collection_job_id": job_id, "datasource_id": source.id, "dataset_id": dataset_id,
                "row_count": collect_result.row_count,
            })
            await db.commit()

        logger.info(f"Collection job {job_id} completed: dataset {dataset_id}, {collect_result.row_count} rows")

        # Auto-trigger characteristics computation for all datasource types that
        # produce numeric time-series data. Skip types whose artifacts are not
        # numeric parquet DataFrames (web_report stores PDFs/HTMLs; economic_calendar
        # has a non-standard schema).
        if source.type not in ("economic_calendar", "web_report"):
            compute_characteristics.apply_async(args=[dataset_id], queue="characteristics")
            logger.info(f"[job {job_id}] auto-enqueued compute_characteristics for dataset {dataset_id}")

        return {"dataset_id": dataset_id, "row_count": collect_result.row_count}
    finally:
        await engine.dispose()
        _release_lock("run_collection_job", job_id)


def _compute_next_run(datasource_type: str, config: dict, schedule_cron: str | None) -> "datetime | None":
    """Return the next UTC datetime this job should run, or None (one-off)."""
    from datetime import timedelta

    # web_report uses interval_days (and optional download_time) from the datasource config
    if datasource_type == "web_report":
        interval_days = config.get("interval_days")
        if interval_days is None:
            return None  # one-off download
        try:
            days = int(interval_days)
        except (TypeError, ValueError):
            return None
        now = datetime.now(timezone.utc)
        download_time = config.get("download_time")  # "HH:MM" UTC, e.g. "18:00"
        if download_time:
            try:
                h, m = map(int, str(download_time).split(":"))
                # Next occurrence of HH:MM that is strictly in the future.
                # Try today first; if that moment has passed, step forward by interval_days.
                candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if candidate <= now:
                    candidate += timedelta(days=days)
                return candidate
            except (ValueError, AttributeError):
                pass  # fall back to interval-only below
        return now + timedelta(days=days)

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
    elif datasource_type == "synthetic_function":
        from data.collectors.synthetic_function import collect
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


class _TrainingResolutionError(Exception):
    """Raised by _resolve_training_context for the not-found cases — caller translates to the
    same {"error": code} shape _train_model has always returned for these."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _json_safe(value):
    """NaN/Inf are valid Python floats but not valid JSON -- Postgres' JSONB column rejects the
    literal "Infinity"/"NaN" tokens asyncpg's encoder produces for them, crashing the whole
    training task on the checkpoint-metrics write (seen: an LSTM run diverging on a token
    representation and writing val_loss=inf). Sanitize right before it goes into a JSONB column;
    the plain float columns (TrainingRun.val_loss, TrainingRunMetric.val_loss) store inf/nan fine.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


async def _resolve_training_context(factory, training_run_id: int):
    """Load the TrainingRun + MLModel, resolve preprocessing (recipe or inline) + the dataset
    artifact, snapshot the resolved hyperparams back onto the run, and flip status to
    running/training. Shared prologue for both the neural (_train_model) and ARIMA
    (_run_arima_training) paths.

    Returns (model_id, architecture, model_config, hp, dataset_artifact, pd_rec).
    Raises _TrainingResolutionError for the not-found cases (run/model/dataset) — matches the
    {"error": ...} returns this prologue always produced, with no status write in those cases
    (pre-existing behavior, unchanged).
    """
    from sqlalchemy import select, update
    from model.models import MLModel, TrainingRun
    from model.architectures import TRAINING_DEFAULTS

    async with factory() as db:
        result = await db.execute(select(TrainingRun).where(TrainingRun.id == training_run_id))
        run = result.scalar_one_or_none()
        if run is None:
            raise _TrainingResolutionError("training_run_not_found")
        result = await db.execute(select(MLModel).where(MLModel.id == run.model_id))
        model_rec = result.scalar_one_or_none()
        if model_rec is None:
            raise _TrainingResolutionError("model_not_found")

        architecture = model_rec.architecture
        model_config = model_rec.config
        dataset_id = run.dataset_id
        hp = {**TRAINING_DEFAULTS.get(architecture, {}), **run.hyperparams}

        # A preprocessed-dataset recipe is the single source of truth for preprocessing/
        # feature_cols/normalize when referenced — overrides any same-named inline hyperparams.
        pd_rec = None
        if run.preprocessed_dataset_id is not None:
            from model.models import PreprocessedDataset
            result = await db.execute(select(PreprocessedDataset).where(PreprocessedDataset.id == run.preprocessed_dataset_id))
            pd_rec = result.scalar_one_or_none()
            if pd_rec is not None:
                hp["preprocessing"] = pd_rec.preprocessing
                hp["feature_cols"] = pd_rec.feature_cols
                hp["normalize"] = pd_rec.normalize

        from data.models import Dataset
        result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
        ds_rec = result.scalar_one_or_none()
        if ds_rec is None or ds_rec.artifact_path is None:
            raise _TrainingResolutionError("dataset_not_found_or_no_artifact")
        dataset_artifact = ds_rec.artifact_path

        # Snapshot the resolved preprocessing config into hyperparams so this run stays
        # fully self-describing even if its recipe is later renamed or deleted.
        snapshot_hp = run.hyperparams
        if pd_rec is not None:
            snapshot_hp = {
                **run.hyperparams,
                "preprocessing": pd_rec.preprocessing,
                "feature_cols": pd_rec.feature_cols,
                "normalize": pd_rec.normalize,
                "preprocessed_dataset_id": pd_rec.id,
                "preprocessed_dataset_name": pd_rec.name,
            }

        await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
            status="running", started_at=datetime.now(timezone.utc), hyperparams=snapshot_hp
        ))
        await db.execute(update(MLModel).where(MLModel.id == model_rec.id).values(status="training"))
        await db.commit()

    return model_rec.id, architecture, model_config, hp, dataset_artifact, pd_rec


async def _run_arima_training(factory, training_run_id: int, model_id: int, architecture: str,
                               model_config: dict, hp: dict, dataset_artifact: str, pd_rec, store: Path) -> dict:
    """AR/MA/ARMA training — a single statsmodels MLE fit, not a torch epoch loop. Called
    inline from _train_model's try/finally, so the outer engine.dispose()/_release_lock still
    covers this path."""
    from sqlalchemy import update
    from model.models import MLModel, TrainingRun, TrainingRunMetric
    from model.trainers import order_from_config, load_series_for_arima, fit_and_evaluate_arima, compute_effective_characteristics

    try:
        order = order_from_config(architecture, model_config)
        train_series, val_series, data_provenance = load_series_for_arima(
            dataset_artifact, hp.get("feature_cols", ["close"]), hp.get("preprocessing"),
            hp.get("normalize", "returns"), hp.get("val_split", 0.2),
        )
        logger.info(f"Training run {training_run_id} data_provenance: {data_provenance}")
        async with factory() as db:
            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                data_provenance=data_provenance
            ))
            await db.commit()
        fit_result = await asyncio.get_event_loop().run_in_executor(
            None, fit_and_evaluate_arima, train_series, val_series, order, hp.get("pred_len", 10)
        )
    except Exception as e:
        async with factory() as db:
            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                status="error", ended_at=datetime.now(timezone.utc)
            ))
            await dispatch(db, "training.error", {
                "training_run_id": training_run_id, "model_id": model_id, "error": str(e),
            })
            await db.commit()
        return {"error": str(e)}

    checkpoint_dir = store / "models" / str(model_id) / f"training_{training_run_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = checkpoint_dir / "best.pkl"
    fit_result["results"].save(str(artifact_path))

    if pd_rec is not None and pd_rec.characteristics is not None:
        preprocessed_characteristics = pd_rec.characteristics
    else:
        try:
            preprocessed_characteristics = compute_effective_characteristics(
                dataset_artifact, hp.get("feature_cols", ["close"]), hp.get("preprocessing")
            )
        except Exception as e:
            preprocessed_characteristics = {"error": str(e)}

    metrics = fit_result["metrics"]
    async with factory() as db:
        db.add(TrainingRunMetric(
            training_run_id=training_run_id, epoch=1,
            train_loss=fit_result["train_mse"], val_loss=metrics["mse"],
        ))
        await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
            status="completed", current_epoch=1, best_epoch=1, val_loss=metrics["mse"],
            num_params=fit_result["n_params"], preprocessed_characteristics=preprocessed_characteristics,
            artifact_path=str(artifact_path.relative_to(store)), ended_at=datetime.now(timezone.utc), eta_seconds=0,
        ))
        await db.execute(update(MLModel).where(MLModel.id == model_id).values(status="trained"))
        await dispatch(db, "training.completed", {
            "training_run_id": training_run_id, "model_id": model_id, "val_loss": metrics["mse"], "best_epoch": 1,
        })
        await db.commit()

    logger.info(f"Training run {training_run_id} ({architecture}) completed. val_loss(mse): {metrics['mse']:.6f}")
    return {"best_epoch": 1, "val_loss": metrics["mse"], "artifact_path": str(artifact_path.relative_to(store))}


async def _train_model(training_run_id: int) -> dict:
    import torch
    import numpy as np
    from sqlalchemy import update
    from model.models import MLModel, TrainingRun, TrainingCheckpoint, TrainingRunMetric
    from model.architectures import build_model
    from model.trainers import get_trainer_fns, get_step_trainer_fn, OHLCWindowDataset, compute_effective_characteristics
    from model.trainers.arima_trainer import ARIMA_ARCHITECTURES
    from celery_app import _get_redis

    # Training tasks can legitimately run for hours, well past a broker's default message
    # visibility window — if the underlying task message gets redelivered while the original
    # execution is still in flight (observed in practice even with a generous
    # broker_transport_options visibility_timeout), a second worker would start an
    # uncoordinated duplicate training loop against the same TrainingRun row. Guard against
    # that directly: only one execution may hold this run's lock at a time.
    redis = _get_redis()
    exec_lock_key = f"algoforge:executing:train_model:{training_run_id}"
    lock_acquired = redis is None or redis.set(exec_lock_key, "1", nx=True, ex=43200)
    if not lock_acquired:
        logger.warning(f"Training run {training_run_id}: duplicate execution detected (already running) — skipping")
        return {"skipped": "duplicate_execution"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = Path(os.getenv("ARTIFACT_STORE_PATH", "artifacts"))

    factory, engine = _make_db()
    try:
        try:
            model_id, architecture, model_config, hp, dataset_artifact, pd_rec = await _resolve_training_context(factory, training_run_id)
        except _TrainingResolutionError as e:
            return {"error": e.code}

        if architecture in ARIMA_ARCHITECTURES:
            return await _run_arima_training(factory, training_run_id, model_id, architecture, model_config, hp, dataset_artifact, pd_rec, store)

        try:
            # Opt-in only. Without it, weight init (and shuffle= ordering) comes from whatever
            # ambient RNG state the worker process happens to be in -- fine normally, but it
            # means two "identical" runs can land 2x+ apart on nothing but init luck (observed
            # live), which makes any comparison claiming a small effect size unreliable unless
            # the seed is controlled and varied deliberately across repeats.
            seed = hp.get("seed")
            if seed is not None:
                import random
                torch.manual_seed(int(seed))
                np.random.seed(int(seed))
                random.seed(int(seed))

            dataset = OHLCWindowDataset(
                dataset_artifact,
                obs_len=hp.get("obs_len", 60),
                pred_len=hp.get("pred_len", 10),
                feature_cols=hp.get("feature_cols", ["close"]),
                normalize=hp.get("normalize", "returns"),
                val_split=hp.get("val_split", 0.2),
                device=device,
                preprocessing=hp.get("preprocessing"),
                max_rows=hp.get("max_rows"),
                token_level=hp.get("token_level"),
                n_bins=hp.get("n_bins", 7),
                cluster_window=hp.get("cluster_window", 20),
                n_clusters=hp.get("n_clusters", 20),
                n_digits=hp.get("n_digits", 3),
                sax_paa_size=hp.get("sax_paa_size", 5),
                tgt_feature_cols=hp.get("tgt_feature_cols"),
                src_normalize=hp.get("src_normalize"),
                split_mode=hp.get("split_mode", "chronological"),
                split_seed=hp.get("split_seed", 42),
                require_contiguous=hp.get("require_contiguous", False),
            )
            # Persisted (not just logged) immediately after construction, before any training
            # happens, so it's visible even if the run later fails or gets orphaned -- exactly
            # the situation that let a silent row-cap truncation go undetected through an entire
            # DDM data-volume investigation phase (see data_provenance's definition).
            logger.info(f"Training run {training_run_id} data_provenance: {dataset.data_provenance}")
            async with factory() as db:
                await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                    data_provenance=dataset.data_provenance
                ))
                await db.commit()
            # Override input_dim/output_dim from actual dataset so the model layer sizes
            # always match the number of selected feature columns. Also sync obs_len/pred_len
            # so architectures that bake those into layer sizes (nbeats, lstm) stay consistent.
            effective_config = {
                **model_config,
                "input_dim": dataset.n_features,
                "output_dim": dataset.n_features,
                "obs_len": hp.get("obs_len", 60),
                "pred_len": hp.get("pred_len", 10),
            }
            # token_level (see OHLCWindowDataset): when set, src is a stream of integer token
            # ids rather than continuous features -- pass the fitted vocab size through so the
            # model builds an embedding front-end instead of its usual continuous input path.
            if dataset.vocab_size is not None:
                effective_config["vocab_size"] = dataset.vocab_size
                effective_config["embedding_dim"] = hp.get("embedding_dim")
            # decoder_only's use_attention=False path (CausalLinearMix) is a fixed-shape layer,
            # unlike attention/LSTM which are sequence-length-agnostic -- it needs the model's
            # true input sequence length at construction time. token_level="digits" expands one
            # time step into (1 + n_digits) token positions, so raw obs_len undercounts it for
            # that case; every other token_level (including None) is 1 token per step.
            tokens_per_step = (1 + dataset.n_digits) if dataset.n_digits is not None else 1
            effective_config["seq_len"] = effective_config["obs_len"] * tokens_per_step
            model = build_model(architecture, effective_config, device=device)
        except Exception as e:
            async with factory() as db:
                await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                    status="error", ended_at=datetime.now(timezone.utc)
                ))
                await dispatch(db, "training.error", {
                    "training_run_id": training_run_id, "model_id": model_id, "error": str(e),
                })
                await db.commit()
            return {"error": str(e)}

        num_params = sum(p.numel() for p in model.parameters())
        if pd_rec is not None and pd_rec.characteristics is not None:
            # Reuse the recipe's cached characteristics (identical inputs) instead of
            # recomputing — falls through to a live compute below if the recipe's background
            # job hasn't finished yet (characteristics still null).
            preprocessed_characteristics = pd_rec.characteristics
        else:
            try:
                preprocessed_characteristics = compute_effective_characteristics(
                    dataset_artifact, hp.get("feature_cols", ["close"]), hp.get("preprocessing"), hp.get("max_rows")
                )
            except Exception as e:
                # Best-effort — never let a characteristics failure abort training.
                preprocessed_characteristics = {"error": str(e)}
        # token_level characteristics are run-specific (depend on this run's hyperparams, not
        # just the recipe), so they're always computed fresh and merged in here -- never cached
        # on pd_rec, unlike preprocessed_characteristics above, which is safe to cache since it
        # only depends on the recipe (dataset/preprocessing/feature_cols).
        if dataset.vocab_size is not None and dataset.token_stream is not None:
            try:
                from model.trainers.dataset import compute_token_characteristics
                preprocessed_characteristics = {
                    **preprocessed_characteristics,
                    **compute_token_characteristics(dataset.token_stream, dataset.vocab_size),
                }
            except Exception as e:
                preprocessed_characteristics = {**preprocessed_characteristics, "token_characteristics_error": str(e)}
        async with factory() as db:
            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                num_params=num_params, preprocessed_characteristics=preprocessed_characteristics
            ))
            await db.commit()

        train_fn, eval_fn = get_trainer_fns(architecture)
        target_lr = hp.get("lr", 0.001)
        # optimizer: opt-in choice of adam (default) / adamw / sgd, to isolate whether Adam's
        # per-parameter moment estimates (m, v) -- not just its epoch-denominated scheduler --
        # are what's driving the data-volume effect. beta1/beta2 override Adam/AdamW's defaults
        # (0.9, 0.999); ignored for sgd. beta1=0 removes momentum entirely.
        # weight_decay: opt-in override, explicit so a comparison across optimizers isn't silently
        # skewed by torch's differing per-optimizer defaults (Adam/SGD default to 0, AdamW to 0.01
        # -- AdamW's whole reason to exist is decoupled decay, so leaving it unset would make an
        # "adam vs adamw" comparison actually a "no decay vs 0.01 decay" comparison instead).
        # None (the default) preserves each optimizer's own torch default unless set explicitly.
        optimizer_name = str(hp.get("optimizer", "adam")).lower()
        beta1 = float(hp.get("beta1", 0.9))
        beta2 = float(hp.get("beta2", 0.999))
        weight_decay = hp.get("weight_decay")
        if optimizer_name == "sgd":
            sgd_kwargs = {"momentum": hp.get("momentum", 0.0)}
            if weight_decay is not None:
                sgd_kwargs["weight_decay"] = float(weight_decay)
            optimizer = torch.optim.SGD(model.parameters(), lr=target_lr, **sgd_kwargs)
        elif optimizer_name == "adamw":
            adamw_kwargs = {"betas": (beta1, beta2)}
            if weight_decay is not None:
                adamw_kwargs["weight_decay"] = float(weight_decay)
            optimizer = torch.optim.AdamW(model.parameters(), lr=target_lr, **adamw_kwargs)
        else:
            adam_kwargs = {"betas": (beta1, beta2)}
            if weight_decay is not None:
                adam_kwargs["weight_decay"] = float(weight_decay)
            optimizer = torch.optim.Adam(model.parameters(), lr=target_lr, **adam_kwargs)
        criterion = None if architecture in ("timegan", "vae") else torch.nn.MSELoss()
        # disable_lr_scheduler: opt-in escape hatch for step-count-controlled comparisons.
        # ReduceLROnPlateau's patience is epoch-denominated same as early_stop_patience, so it's
        # a second epoch-length-dependent confound that survives even when early stopping is
        # disabled -- isolating "does optimizer step count alone explain the data-volume effect"
        # requires taking this out of the picture too, not just running a fixed epoch count.
        disable_lr_scheduler = bool(hp.get("disable_lr_scheduler", False))
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            if criterion and not disable_lr_scheduler else None
        )

        epochs = int(hp.get("epochs", 50))
        batch_size = int(hp.get("batch_size", 32))
        shuffle = bool(hp.get("shuffle", False))
        # Opt-in only -- never inferred from lr/batch_size. Linear ramp from lr/N up to the
        # full target lr over the first N epochs; the plateau scheduler is held off until
        # warmup finishes so it doesn't fight the ramp with its own reductions.
        lr_warmup_epochs = int(hp.get("lr_warmup_epochs", 0) or 0)
        early_stop_patience = hp.get("early_stop_patience")
        early_stop_patience = int(early_stop_patience) if early_stop_patience else None
        divergence_factor = hp.get("divergence_factor")
        divergence_factor = float(divergence_factor) if divergence_factor else None
        epochs_since_improvement = 0
        best_val_loss = float("inf")
        best_epoch = 0
        checkpoint_dir = store / "models" / str(model_id) / f"training_{training_run_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # max_steps: opt-in escape hatch that removes the concept of "epoch" from the loop
        # entirely -- an infinite, reshuffled stream of training windows (train_steps), with
        # validation/checkpointing/early-stopping keyed to a step counter (val_every_steps) that
        # has no relationship to how many windows one pass through the data takes. Exists because
        # even with epochs run to a fixed step count and the LR scheduler disabled (see
        # disable_lr_scheduler), a residual gap between many-small-epochs and few-huge-epochs
        # training survived -- this tests whether *any* periodic epoch-boundary structure
        # (validation timing, early-stop checks) is still contributing, or whether it's purely
        # about the batch sequence / optimizer state itself. epoch-keyed features above
        # (lr_warmup_epochs, the ReduceLROnPlateau scheduler) don't translate to step space and
        # are not applied in this mode -- pair max_steps with disable_lr_scheduler=true.
        max_steps = hp.get("max_steps")
        loop = asyncio.get_event_loop()
        if max_steps is not None:
            max_steps = int(max_steps)
            val_every_steps = int(hp.get("val_every_steps", 5000))
            # Patience counted in validation *checks*, not epochs -- a check happens every
            # val_every_steps regardless of dataset size, so this is already step-denominated.
            early_stop_patience_checks = hp.get("early_stop_patience_checks")
            early_stop_patience_checks = int(early_stop_patience_checks) if early_stop_patience_checks else None
            step_train_fn = get_step_trainer_fn(architecture)
            steps_done = 0
            n_checks = 0
            checks_since_improvement = 0
            while steps_done < max_steps:
                async with factory() as db:
                    stop_result = await db.execute(
                        __import__("sqlalchemy", fromlist=["select"]).select(
                            TrainingRun.stop_requested
                        ).where(TrainingRun.id == training_run_id)
                    )
                    if stop_result.scalar():
                        logger.info(f"Training run {training_run_id} stopped at step {steps_done}")
                        break

                chunk = min(val_every_steps, max_steps - steps_done)
                train_loss = await loop.run_in_executor(
                    None, step_train_fn, model, dataset, optimizer, criterion, batch_size, chunk
                )
                steps_done += chunk
                n_checks += 1
                val_loss = await loop.run_in_executor(None, eval_fn, model, dataset, criterion, batch_size)

                # n_checks is stored in the TrainingCheckpoint/TrainingRunMetric "epoch" column
                # (no schema change for what's fundamentally a research-mode loop) -- multiply by
                # val_every_steps (recorded in hyperparams) to recover the actual step count.
                ckpt_path = checkpoint_dir / f"epoch_{n_checks:04d}.pt"
                torch.save({"epoch": n_checks, "step": steps_done, "model_state": model.state_dict(), "val_loss": val_loss}, ckpt_path)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = n_checks
                    checks_since_improvement = 0
                    torch.save({"epoch": n_checks, "step": steps_done, "model_state": model.state_dict(), "val_loss": val_loss}, checkpoint_dir / "best.pt")
                else:
                    checks_since_improvement += 1

                eta_seconds = 0

                async with factory() as db:
                    await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                        current_epoch=n_checks, val_loss=val_loss, best_epoch=best_epoch, eta_seconds=eta_seconds
                    ))
                    db.add(TrainingCheckpoint(
                        training_run_id=training_run_id,
                        epoch=n_checks,
                        metrics={"train_loss": _json_safe(train_loss), "val_loss": _json_safe(val_loss), "step": steps_done},
                        artifact_path=str(ckpt_path.relative_to(store)),
                    ))
                    db.add(TrainingRunMetric(
                        training_run_id=training_run_id,
                        epoch=n_checks,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        lr=optimizer.param_groups[0].get("lr"),
                    ))
                    await db.commit()

                logger.info(
                    f"Training run {training_run_id} step {steps_done}/{max_steps} "
                    f"(check {n_checks}): train={train_loss:.6f} val={val_loss:.6f}"
                )

                if early_stop_patience_checks is not None and checks_since_improvement >= early_stop_patience_checks:
                    logger.info(
                        f"Training run {training_run_id} early-stopped at step {steps_done} "
                        f"(no improvement for {early_stop_patience_checks} checks, best={best_val_loss:.6f} @ check {best_epoch})"
                    )
                    break

                if divergence_factor is not None and val_loss > best_val_loss * divergence_factor:
                    logger.info(
                        f"Training run {training_run_id} diverged at step {steps_done}: "
                        f"val={val_loss:.6f} exceeds {divergence_factor}x best ({best_val_loss:.6f}) — stopping"
                    )
                    break
        else:
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

                if lr_warmup_epochs > 0 and epoch <= lr_warmup_epochs:
                    warmup_lr = target_lr * epoch / lr_warmup_epochs
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr

                train_loss = await loop.run_in_executor(None, train_fn, model, dataset, optimizer, criterion, batch_size, shuffle)
                val_loss = await loop.run_in_executor(None, eval_fn, model, dataset, criterion, batch_size)

                if scheduler and epoch >= lr_warmup_epochs:
                    scheduler.step(val_loss)

                ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss}, ckpt_path)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_since_improvement = 0
                    torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_loss": val_loss}, checkpoint_dir / "best.pt")
                else:
                    epochs_since_improvement += 1

                eta_seconds = (epochs - epoch) * 5

                async with factory() as db:
                    await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                        current_epoch=epoch, val_loss=val_loss, best_epoch=best_epoch, eta_seconds=eta_seconds
                    ))
                    db.add(TrainingCheckpoint(
                        training_run_id=training_run_id,
                        epoch=epoch,
                        metrics={"train_loss": _json_safe(train_loss), "val_loss": _json_safe(val_loss)},
                        artifact_path=str(ckpt_path.relative_to(store)),
                    ))
                    db.add(TrainingRunMetric(
                        training_run_id=training_run_id,
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        lr=optimizer.param_groups[0].get("lr"),
                    ))
                    await db.commit()

                logger.info(f"Training run {training_run_id} epoch {epoch}/{epochs}: train={train_loss:.6f} val={val_loss:.6f}")

                if early_stop_patience is not None and epochs_since_improvement >= early_stop_patience:
                    logger.info(
                        f"Training run {training_run_id} early-stopped at epoch {epoch} "
                        f"(no improvement for {early_stop_patience} epochs, best={best_val_loss:.6f} @ epoch {best_epoch})"
                    )
                    break

                # Divergence stop: distinct from patience above. Patience fires on a *plateau*
                # (many epochs with no improvement, however small the gap); this fires the moment
                # val_loss gets much *worse* than the best seen so far, however few epochs that
                # takes — catches a blown-up run (bad batch_size/lr interaction, NaN-adjacent
                # instability) long before patience would.
                if divergence_factor is not None and val_loss > best_val_loss * divergence_factor:
                    logger.info(
                        f"Training run {training_run_id} diverged at epoch {epoch}: "
                        f"val={val_loss:.6f} exceeds {divergence_factor}x best ({best_val_loss:.6f}) — stopping"
                    )
                    break

        best_artifact = str((checkpoint_dir / "best.pt").relative_to(store))
        async with factory() as db:
            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                status="completed", ended_at=datetime.now(timezone.utc),
                best_epoch=best_epoch, val_loss=best_val_loss,
                artifact_path=best_artifact, eta_seconds=0,
            ))
            await db.execute(update(MLModel).where(MLModel.id == model_id).values(status="trained"))
            await dispatch(db, "training.completed", {
                "training_run_id": training_run_id, "model_id": model_id,
                "val_loss": best_val_loss, "best_epoch": best_epoch,
            })
            await db.commit()

        logger.info(f"Training run {training_run_id} completed. Best epoch: {best_epoch}, val_loss: {best_val_loss:.6f}")
        return {"best_epoch": best_epoch, "val_loss": best_val_loss, "artifact_path": best_artifact}
    except Exception as e:
        # Catch-all for anything that escapes the training loop itself (as opposed to the
        # narrower try/except around model construction above) -- without this, an exception
        # raised mid-training (e.g. an architecture-specific ValueError like PairLagModel's
        # pool_size-vs-obs_len check) leaves the TrainingRun permanently stuck at status
        # "running": Celery logs the task as failed, but nothing ever updates the DB row, so it
        # never shows up as an error to poll against and blocks any "all done" check keyed off
        # pending/running counts. Observed live during the lag-distance sweep (2026-08-26).
        logger.exception(f"Training run {training_run_id} failed during training")
        async with factory() as db:
            await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id).values(
                status="error", ended_at=datetime.now(timezone.utc)
            ))
            await dispatch(db, "training.error", {
                "training_run_id": training_run_id, "model_id": model_id, "error": str(e),
            })
            await db.commit()
        return {"error": str(e)}
    finally:
        await engine.dispose()
        _release_lock("train_model", training_run_id)
        if redis is not None:
            redis.delete(exec_lock_key)


# ---------------------------------------------------------------------------
# Task 3b — colab_train_model (execution_target="colab" — see model/colab_trainer.py)
# ---------------------------------------------------------------------------

@celery_app.task(name="celery_worker.colab_train_model", bind=False)
def colab_train_model(training_run_id: int) -> dict:
    from model.colab_trainer import run_colab_training
    return asyncio.run(run_colab_training(training_run_id))


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


# ---------------------------------------------------------------------------
# Task 6 — compute_preprocessed_characteristics
# ---------------------------------------------------------------------------

@celery_app.task(name="celery_worker.compute_preprocessed_characteristics", bind=False)
def compute_preprocessed_characteristics(preprocessed_dataset_id: int) -> dict:
    return asyncio.run(_compute_preprocessed_characteristics(preprocessed_dataset_id))


async def _compute_preprocessed_characteristics(preprocessed_dataset_id: int) -> dict:
    from sqlalchemy import select, update
    from model.models import PreprocessedDataset
    from data.models import Dataset
    from model.trainers.dataset import compute_effective_characteristics

    logger.info(f"compute_preprocessed_characteristics started for {preprocessed_dataset_id}")
    factory, engine = _make_db()
    try:
        async with factory() as db:
            result = await db.execute(select(PreprocessedDataset).where(PreprocessedDataset.id == preprocessed_dataset_id))
            pd_rec = result.scalar_one_or_none()
            if pd_rec is None:
                return {"error": "preprocessed_dataset_not_found"}
            result = await db.execute(select(Dataset).where(Dataset.id == pd_rec.dataset_id))
            ds_rec = result.scalar_one_or_none()
            if ds_rec is None or ds_rec.artifact_path is None:
                async with factory() as db2:
                    await db2.execute(update(PreprocessedDataset).where(PreprocessedDataset.id == preprocessed_dataset_id).values(status="error"))
                    await db2.commit()
                return {"error": "dataset_not_found_or_no_artifact"}
            dataset_artifact = ds_rec.artifact_path
            preprocessing = pd_rec.preprocessing
            feature_cols = pd_rec.feature_cols

        try:
            characteristics = await asyncio.to_thread(compute_effective_characteristics, dataset_artifact, feature_cols, preprocessing)
            status = "ready"
        except Exception as e:
            logger.exception(f"compute_preprocessed_characteristics failed for {preprocessed_dataset_id}")
            characteristics = {"error": str(e)}
            status = "error"

        async with factory() as db:
            await db.execute(update(PreprocessedDataset).where(PreprocessedDataset.id == preprocessed_dataset_id).values(
                characteristics=characteristics, status=status,
            ))
            await db.commit()

        logger.info(f"compute_preprocessed_characteristics done for {preprocessed_dataset_id}: status={status}")
        return {"status": status}
    finally:
        await engine.dispose()
        _release_lock("compute_preprocessed_characteristics", preprocessed_dataset_id)
