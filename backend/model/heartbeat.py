"""Training-run liveness (requirements.md R-13).

A training run whose worker process dies (OOM kill, restart, a human stopping the worker) used to
stay `running` forever: status transitions and the `training.error` webhook are only ever written
by the executing worker itself, and `stop_training_run` only sets a flag that the (dead) training
loop would have read. An autonomous caller could not tell "still training" from "worker is gone".

Three pieces:

- HeartbeatThread: started by the worker's train_model task for the whole execution. Writes
  TrainingRun.last_heartbeat_at every HEARTBEAT_INTERVAL_SECONDS from its own thread, event loop
  and DB engine -- so it keeps beating even while the task's event loop is blocked by synchronous
  work (dataset construction, characteristics), and stops only when the process does. It
  measures "is the process alive", not "is training making progress".
- reap_stale_training_runs(): marks `running` runs whose heartbeat is older than
  STALE_AFTER_SECONDS as `error` with an error_message, dispatches `training.error`, and clears the
  run's Redis locks. Runs with NO heartbeat at all (started by a worker that predates this code,
  or legacy rows) are never touched -- staleness can only be judged against a heartbeat that was
  actually being written. The UPDATE is conditional on the same staleness predicate, so several
  reapers (e.g. multiple API replicas) cannot double-process a run.
- reaper_loop(): calls the reaper periodically; started from the API's lifespan (main.py), since
  the API is the one process that is always up (Celery Beat is not running in every deployment).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, select, update

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = float(os.getenv("ALGOFORGE_HEARTBEAT_INTERVAL_SECONDS", "60"))
STALE_AFTER_SECONDS = float(os.getenv("ALGOFORGE_HEARTBEAT_STALE_SECONDS", "600"))
REAPER_INTERVAL_SECONDS = float(os.getenv("ALGOFORGE_REAPER_INTERVAL_SECONDS", "120"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime | None) -> datetime | None:
    """Postgres returns aware timestamps; SQLite (tests) returns naive ones, stored as UTC."""
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def is_heartbeat_stale(last_heartbeat_at: datetime | None, now: datetime | None = None,
                       stale_after: float = STALE_AFTER_SECONDS) -> bool | None:
    """True/False for a run with a heartbeat; None when there is no heartbeat to judge."""
    if last_heartbeat_at is None:
        return None
    return ((now or _now()) - as_utc(last_heartbeat_at)).total_seconds() > stale_after


async def write_heartbeat(factory, training_run_id: int) -> None:
    from model.models import TrainingRun

    async with factory() as db:
        await db.execute(update(TrainingRun).where(TrainingRun.id == training_run_id)
                         .values(last_heartbeat_at=_now()))
        await db.commit()


class HeartbeatThread:
    """Daemon thread writing a run's heartbeat until stop(). make_db() must return a fresh
    (session_factory, engine) pair -- the thread owns its engine, since an asyncio engine is bound
    to the event loop it was created on."""

    def __init__(self, training_run_id: int, make_db, interval: float = HEARTBEAT_INTERVAL_SECONDS):
        self.training_run_id = training_run_id
        self._make_db = make_db
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{training_run_id}", daemon=True)

    def start(self) -> "HeartbeatThread":
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
                    await write_heartbeat(factory, self.training_run_id)
                except Exception as e:  # noqa: BLE001 -- a DB hiccup must never fail the run
                    logger.warning(f"Training run {self.training_run_id}: heartbeat write failed: {e}")
                # wait on the stop event without blocking this thread's loop forever
                await asyncio.get_running_loop().run_in_executor(None, self._stop.wait, self._interval)
        finally:
            await engine.dispose()


def _clear_run_locks(training_run_id: int) -> None:
    """Drop the run's enqueue-dedup and execution locks so a retry is not skipped as a duplicate.
    Mirrors celery_worker._release_lock and train_model's exec lock key."""
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        r.delete(f"algoforge:enqueued:train_model:{training_run_id}",
                 f"algoforge:executing:train_model:{training_run_id}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Training run {training_run_id}: could not clear Redis locks: {e}")


async def mark_worker_lost(db, run, reason: str) -> bool:
    """Move one run to `error` if it is still `running`; dispatch training.error. Returns whether
    this call made the transition (False if another process already did)."""
    from model.models import TrainingRun
    from webhooks.dispatcher import dispatch

    result = await db.execute(
        update(TrainingRun)
        .where(and_(TrainingRun.id == run.id, TrainingRun.status == "running"))
        .values(status="error", error_message=reason, ended_at=_now())
    )
    if result.rowcount != 1:
        return False
    await dispatch(db, "training.error", {
        "training_run_id": run.id, "model_id": run.model_id, "error": reason,
        "error_code": "worker_lost",
    })
    return True


async def reap_stale_training_runs(db, now: datetime | None = None,
                                   stale_after: float = STALE_AFTER_SECONDS) -> list[int]:
    """Reap `running` runs whose heartbeat is older than `stale_after` seconds. The caller commits."""
    from model.models import TrainingRun

    now = now or _now()
    cutoff = now - timedelta(seconds=stale_after)
    stale = (await db.execute(select(TrainingRun).where(and_(
        TrainingRun.status == "running",
        TrainingRun.last_heartbeat_at.isnot(None),
        TrainingRun.last_heartbeat_at < cutoff,
    )))).scalars().all()
    reaped = []
    for run in stale:
        reason = (f"worker lost: no heartbeat since {as_utc(run.last_heartbeat_at).isoformat()} "
                  f"(stale after {int(stale_after)}s)")
        if await mark_worker_lost(db, run, reason):
            reaped.append(run.id)
    for run_id in reaped:
        _clear_run_locks(run_id)
    if reaped:
        logger.warning(f"Reaped training runs with stale heartbeats: {reaped}")
    return reaped


async def reaper_loop(interval: float = REAPER_INTERVAL_SECONDS) -> None:
    from database import db_session

    while True:
        try:
            async with db_session() as db:
                await reap_stale_training_runs(db)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Training-run reaper pass failed: {e}")
        await asyncio.sleep(interval)
