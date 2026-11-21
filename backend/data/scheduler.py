"""Collection-job scheduling and liveness.

Why this module exists (web_report jobs "never download" / "stop after one run"):

- The scheduler tick used to run as a Celery Beat task on the `collection` queue. With the local
  `--pool=solo` workers, a tick waits behind whatever collection job is running (an endless DDM job
  starves it forever), and it ran nowhere at all when Beat wasn't started. It now runs in the API
  process (scheduler_loop, started from main.py's lifespan), like the training-run reaper.
- next_run_at was only written after a *successful* run: a new job (NULL) was never picked up, and a
  job whose run errored kept its past next_run_at, so every tick re-enqueued it (an error-retry
  storm that occupied the collection worker). Now:
    * a schedulable job with NULL next_run_at is backfilled (due now if it has never run),
    * the tick advances next_run_at as soon as it enqueues, so a job is enqueued once per period,
    * failures set next_run_at = retry_run_at() (bounded retry instead of every minute).
- A job whose worker died (reboot, Ctrl+C, hung browser) stayed `running` forever, and the tick
  skips `running` jobs. Workers now write CollectionJob.last_heartbeat_at (CollectionHeartbeat), and
  reap_stale_collection_jobs() moves stale `running` jobs to `error` and schedules a retry. Jobs
  with no heartbeat (started by a worker predating this code) are reaped only when their Redis
  enqueue lock -- which every live run_collection_job renews -- is gone.
- The tick never deletes the enqueue dedup lock any more (it used to, which let the same job be
  enqueued again on every tick while it waited in the queue).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from sqlalchemy import and_, select, update

logger = logging.getLogger(__name__)

SCHEDULER_INTERVAL_SECONDS = float(os.getenv("ALGOFORGE_SCHEDULER_INTERVAL_SECONDS", "60"))
RETRY_MINUTES = float(os.getenv("ALGOFORGE_COLLECTION_RETRY_MINUTES", "30"))
HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("ALGOFORGE_COLLECTION_HEARTBEAT_SECONDS", "60"))
STALE_AFTER_SECONDS = float(os.getenv("ALGOFORGE_COLLECTION_STALE_SECONDS", "600"))

ENQUEUE_LOCK_KEY = "algoforge:enqueued:run_collection_job:{job_id}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime | None) -> datetime | None:
    """Postgres returns aware timestamps; SQLite (tests) returns naive ones, stored as UTC."""
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# next_run_at computation
# ---------------------------------------------------------------------------

def compute_next_run(datasource_type: str, config: dict | None, schedule_cron: str | None,
                     now: datetime | None = None) -> datetime | None:
    """Return the next UTC datetime this job should run, or None (one-off)."""
    now = now or _now()
    config = config or {}

    # web_report uses interval_days (and optional download_time "HH:MM" UTC) from the datasource config
    if datasource_type == "web_report":
        interval_days = config.get("interval_days")
        if interval_days is None:
            return None  # one-off download
        try:
            days = int(interval_days)
        except (TypeError, ValueError):
            return None
        if days <= 0:
            return None
        download_time = config.get("download_time")
        if download_time:
            try:
                h, m = map(int, str(download_time).split(":"))
                # Next occurrence of HH:MM that is strictly in the future.
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
            return trigger.get_next_fire_time(None, now)
        except Exception:
            logger.warning(f"compute_next_run: invalid schedule_cron {schedule_cron!r}")

    return None


def retry_run_at(datasource_type: str, config: dict | None, schedule_cron: str | None,
                 now: datetime | None = None, retry_minutes: float = RETRY_MINUTES) -> datetime | None:
    """next_run_at after a failed run: retry after `retry_minutes`, but never later than the regular
    next run. One-off jobs (no schedule) are not retried automatically."""
    now = now or _now()
    regular = compute_next_run(datasource_type, config, schedule_cron, now)
    if regular is None:
        return None
    return min(regular, now + timedelta(minutes=retry_minutes))


# ---------------------------------------------------------------------------
# Redis helpers (sync; call through asyncio.to_thread from async code)
# ---------------------------------------------------------------------------

def _redis():
    import redis as _r
    return _r.from_url(os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"), decode_responses=True,
                       socket_connect_timeout=5, socket_timeout=5)


def enqueue_lock_held(job_id: int) -> bool | None:
    """True/False whether run_collection_job's enqueue lock exists; None if Redis is unreachable."""
    if os.getenv("ALGOFORGE_NO_REDIS", "").lower() in ("1", "true"):
        return None
    try:
        return bool(_redis().exists(ENQUEUE_LOCK_KEY.format(job_id=job_id)))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"collection scheduler: Redis unavailable ({e})")
        return None


def release_enqueue_lock(job_id: int) -> None:
    try:
        _redis().delete(ENQUEUE_LOCK_KEY.format(job_id=job_id))
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Scheduler tick
# ---------------------------------------------------------------------------

async def _default_enqueue(job_id: int) -> None:
    from celery_app import enqueue
    await enqueue("run_collection_job", job_id)


async def _default_lock_held(job_id: int) -> bool | None:
    return await asyncio.to_thread(enqueue_lock_held, job_id)


async def scheduler_tick(
    db,
    now: datetime | None = None,
    *,
    enqueue: Callable[[int], Awaitable[None]] = _default_enqueue,
    lock_held: Callable[[int], Awaitable[bool | None]] = _default_lock_held,
) -> list[int]:
    """Backfill missing next_run_at and enqueue due jobs. Returns enqueued job ids. The caller commits."""
    from celery_app import AlreadyRunningError
    from data.models import CollectionJob, Datasource

    now = now or _now()
    rows = (await db.execute(
        select(CollectionJob, Datasource)
        .join(Datasource, Datasource.id == CollectionJob.datasource_id)
        .where(CollectionJob.enabled == True, CollectionJob.status != "running")  # noqa: E712
    )).all()

    enqueued: list[int] = []
    for job, source in rows:
        next_run_at = as_utc(job.next_run_at)

        if next_run_at is None:
            regular = compute_next_run(source.type, source.config, job.schedule_cron, now)
            if regular is None:
                continue  # one-off job: only runs when triggered manually
            # Never run (or last attempt left no schedule behind): due now; otherwise regular slot.
            next_run_at = now if (job.last_run_at is None or job.status == "error") else regular
            await db.execute(update(CollectionJob).where(CollectionJob.id == job.id)
                             .values(next_run_at=next_run_at))
            logger.info(f"[scheduler] job {job.id}: backfilled next_run_at={next_run_at.isoformat()}")

        if next_run_at > now:
            continue
        if await lock_held(job.id):
            continue  # already queued / running

        try:
            await enqueue(job.id)
        except AlreadyRunningError:
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[scheduler] could not enqueue job {job.id}: {exc}")
            continue

        # Advance right away so the job is enqueued once per period even if the run never reports
        # back. The worker rewrites next_run_at when it finishes (success or retry on failure).
        following = compute_next_run(source.type, source.config, job.schedule_cron, now)
        await db.execute(update(CollectionJob).where(CollectionJob.id == job.id)
                         .values(next_run_at=following))
        enqueued.append(job.id)
        logger.info(f"[scheduler] enqueued job {job.id} (was due {next_run_at.isoformat()}, "
                    f"next {following.isoformat() if following else 'none'})")
    return enqueued


# ---------------------------------------------------------------------------
# Liveness: heartbeat + reaper
# ---------------------------------------------------------------------------

async def write_heartbeat(factory, job_id: int) -> None:
    from data.models import CollectionJob

    async with factory() as db:
        await db.execute(update(CollectionJob).where(CollectionJob.id == job_id)
                         .values(last_heartbeat_at=_now()))
        await db.commit()


class CollectionHeartbeat:
    """Daemon thread writing a collection job's heartbeat until stop(). make_db() must return a
    fresh (session_factory, engine) pair -- the thread owns its engine and event loop, so it keeps
    beating while the task's own thread is blocked in a synchronous collector."""

    def __init__(self, job_id: int, make_db, interval: float = HEARTBEAT_INTERVAL_SECONDS):
        self.job_id = job_id
        self._make_db = make_db
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"collect-heartbeat-{job_id}", daemon=True)

    def start(self) -> "CollectionHeartbeat":
        self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        asyncio.run(self._beat())

    async def _beat(self) -> None:
        factory, engine = self._make_db()
        try:
            while not self._stop.is_set():
                try:
                    await write_heartbeat(factory, self.job_id)
                except Exception as e:  # noqa: BLE001 -- a DB hiccup must never fail the job
                    logger.warning(f"Collection job {self.job_id}: heartbeat write failed: {e}")
                await asyncio.get_running_loop().run_in_executor(None, self._stop.wait, self._interval)
        finally:
            await engine.dispose()


async def reap_stale_collection_jobs(
    db,
    now: datetime | None = None,
    stale_after: float = STALE_AFTER_SECONDS,
    *,
    lock_held: Callable[[int], Awaitable[bool | None]] = _default_lock_held,
    release_lock: Callable[[int], None] = release_enqueue_lock,
) -> list[int]:
    """Move `running` jobs whose worker is gone to `error` and schedule a retry. The caller commits."""
    from data.models import CollectionJob, Datasource
    from webhooks.dispatcher import dispatch

    now = now or _now()
    cutoff = now - timedelta(seconds=stale_after)
    rows = (await db.execute(
        select(CollectionJob, Datasource)
        .join(Datasource, Datasource.id == CollectionJob.datasource_id)
        .where(CollectionJob.status == "running")
    )).all()

    reaped: list[int] = []
    for job, source in rows:
        hb = as_utc(job.last_heartbeat_at)
        if hb is not None:
            if hb >= cutoff:
                continue
            reason = f"worker lost: no heartbeat since {hb.isoformat()} (stale after {int(stale_after)}s)"
            stale_pred = and_(CollectionJob.last_heartbeat_at.isnot(None), CollectionJob.last_heartbeat_at < cutoff)
        else:
            # Legacy worker without heartbeats: it still renews the enqueue lock while alive.
            if await lock_held(job.id) is not False:
                continue  # alive, or Redis unknown -> don't judge
            reason = "worker lost: job was running but its worker no longer holds the queue lock"
            stale_pred = CollectionJob.last_heartbeat_at.is_(None)

        result = await db.execute(
            update(CollectionJob)
            .where(and_(CollectionJob.id == job.id, CollectionJob.status == "running"))
            .values(status="error", last_error=reason, last_run_at=now,
                    next_run_at=retry_run_at(source.type, source.config, job.schedule_cron, now))
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            continue
        await dispatch(db, "collection.error", {
            "collection_job_id": job.id, "datasource_id": source.id, "error": reason,
            "error_code": "worker_lost",
        })
        release_lock(job.id)
        reaped.append(job.id)

    if reaped:
        logger.warning(f"Reaped collection jobs whose worker was lost: {reaped}")
    return reaped


async def scheduler_loop(interval: float = SCHEDULER_INTERVAL_SECONDS) -> None:
    """Reap dead jobs, then enqueue due ones -- every `interval` seconds. Runs in the API process."""
    from database import db_session

    while True:
        try:
            async with db_session() as db:
                await reap_stale_collection_jobs(db)
            async with db_session() as db:
                await scheduler_tick(db)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Collection scheduler pass failed: {e}")
        await asyncio.sleep(interval)
