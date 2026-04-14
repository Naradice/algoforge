"""
Celery application instance and enqueue helper.

Workers are started per queue:
    celery -A celery_worker worker -Q collection      -c 3  --pool=prefork
    celery -A celery_worker worker -Q characteristics -c 12 --pool=prefork
    celery -A celery_worker worker -Q training        -c 2  --pool=prefork
    celery -A celery_worker worker -Q backtest        -c 5  --pool=prefork
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Path bootstrap — resolve() gives a canonical absolute path on all platforms.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from celery import Celery

logger = logging.getLogger("celery_app")

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_NO_REDIS = os.getenv("ALGOFORGE_NO_REDIS", "").lower() in ("1", "true")

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

celery_app = Celery(
    "algoforge",
    broker=_REDIS_URL,
    backend=_REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    # Don't pre-fetch tasks — each prefork worker handles one heavy job at a time.
    worker_prefetch_multiplier=1,
    # Ack only after the task completes so a killed worker re-queues the task.
    task_acks_late=True,
    # Keep results long enough for dedup checks (24 h).
    result_expires=86400,
    task_routes={
        "celery_worker.run_collection_job":    {"queue": "collection"},
        "celery_worker.compute_characteristics": {"queue": "characteristics"},
        "celery_worker.train_model":           {"queue": "training"},
        "celery_worker.validate_model":        {"queue": "training"},
        "celery_worker.execute_strategy_run":  {"queue": "backtest"},
        "celery_worker.tick_scheduler":        {"queue": "collection"},
    },
    # Beat schedule — tick_scheduler fires every 60 s and enqueues any due collection jobs.
    beat_schedule={
        "tick-scheduler": {
            "task": "celery_worker.tick_scheduler",
            "schedule": 60.0,
        },
    },
    timezone="UTC",
)

# ---------------------------------------------------------------------------
# Redis client for dedup lock (separate from Celery broker connection)
# ---------------------------------------------------------------------------

_redis_client: Any = None


def _get_redis() -> Any | None:
    global _redis_client
    if _NO_REDIS:
        return None
    if _redis_client is None:
        try:
            import redis as _redis
            _redis_client = _redis.from_url(_REDIS_URL, decode_responses=True)
        except Exception:
            logger.warning("Failed to create Redis dedup client", exc_info=True)
    return _redis_client


# ---------------------------------------------------------------------------
# enqueue() — drop-in replacement for arq_pool.enqueue()
# ---------------------------------------------------------------------------

class AlreadyRunningError(RuntimeError):
    """Raised when a dedup lock is already held for this entity."""


async def enqueue(task_name: str, *args) -> None:
    """Enqueue a background task.

    With Redis: uses a deterministic Redis lock key for dedup, but assigns a
      fresh UUID as the Celery task_id each time.  Using a fixed task_id caused
      previously-revoked tasks to be permanently blacklisted by Celery, making
      the job un-runnable until the worker restarted.

    Without Redis (ALGOFORGE_NO_REDIS=1): runs the Celery task synchronously
      in a thread pool so the FastAPI event loop is not blocked.

    Raises AlreadyRunningError if the dedup lock is already held.
    """
    import uuid
    lock_id = f"{task_name}:{args[0] if args else 'noarg'}"

    if _NO_REDIS:
        # Inline fallback: run the sync Celery task in a thread.
        import celery_worker as _w
        task_fn = getattr(_w, task_name)
        logger.warning(f"enqueue → inline (no Redis): {task_name}({args})")
        await asyncio.to_thread(task_fn, *args)
        return

    # Dedup: use Redis SET NX so a second enqueue for the same entity while the
    # first is still running is a visible error (not a silent drop).
    redis = _get_redis()
    if redis is not None:
        lock_key = f"algoforge:enqueued:{lock_id}"
        acquired = redis.set(lock_key, "1", nx=True, ex=3600)
        if not acquired:
            logger.warning(f"enqueue: lock already held for {lock_id}")
            raise AlreadyRunningError(
                f"A {task_name} job is already queued or running. "
                "Wait for it to finish, or run scripts/clear-stuck-jobs.bat if it is stuck."
            )

    import celery_worker as _w
    task_fn = getattr(_w, task_name)
    entity_id = args[0] if args else "noarg"
    # Use a fresh UUID so a previously-revoked deterministic ID never blocks re-runs.
    celery_task_id = str(uuid.uuid4())
    logger.info(f"enqueue → Redis: {task_name}({entity_id})  celery_task_id={celery_task_id}")
    task_fn.apply_async(args=list(args), task_id=celery_task_id)
