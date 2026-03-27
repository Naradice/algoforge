"""
Async batch writer for the `logs` table.

Log entries go into an in-process asyncio.Queue.  A background task drains
the queue in batches (flush every 50 ms or when 200 rows accumulate).

DEBUG entries are dropped under backpressure (queue full).
ERROR / CRITICAL entries bypass the queue and are written immediately.

Start the writer at app startup:
    from .log_writer import start_log_writer
    asyncio.create_task(start_log_writer())
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from .database import async_session_factory

_FLUSH_INTERVAL = 0.05  # seconds
_BATCH_SIZE = 200
_QUEUE_MAX = 5_000

_queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
_writer_task: asyncio.Task | None = None


def enqueue_log(
    *,
    level: str,
    source: str,
    message: str,
    context: dict[str, Any] | None,
    strategy_run_id: int | None,
    training_run_id: int | None,
    collection_job_id: int | None,
    event_id: int | None,
) -> None:
    """Non-blocking enqueue. Drops DEBUG entries when the queue is full."""
    entry = {
        "created_at": datetime.now(timezone.utc),
        "level": level,
        "source": source,
        "message": message,
        "context": context,
        "strategy_run_id": strategy_run_id,
        "training_run_id": training_run_id,
        "collection_job_id": collection_job_id,
        "event_id": event_id,
    }
    try:
        _queue.put_nowait(entry)
    except asyncio.QueueFull:
        if level in ("ERROR", "CRITICAL"):
            # Schedule an immediate write for high-priority entries even under pressure.
            asyncio.create_task(_write_immediate(entry))
        # DEBUG / INFO / WARNING are silently dropped when queue is full.


async def _write_immediate(entry: dict) -> None:
    await _flush_batch([entry])


async def _flush_batch(entries: list[dict]) -> None:
    if not entries:
        return
    async with async_session_factory() as session:
        await session.execute(
            sa.text(
                """
                INSERT INTO logs
                    (created_at, level, source, message, context,
                     strategy_run_id, training_run_id, collection_job_id, event_id)
                VALUES
                    (:created_at, :level, :source, :message, :context::jsonb,
                     :strategy_run_id, :training_run_id, :collection_job_id, :event_id)
                """
            ),
            entries,
        )
        await session.commit()


async def start_log_writer() -> None:
    """Long-running background coroutine — run as an asyncio Task at startup."""
    global _writer_task
    _writer_task = asyncio.current_task()
    batch: list[dict] = []

    while True:
        # Collect for up to _FLUSH_INTERVAL seconds or until batch is full
        deadline = asyncio.get_event_loop().time() + _FLUSH_INTERVAL
        while len(batch) < _BATCH_SIZE:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                entry = await asyncio.wait_for(_queue.get(), timeout=remaining)
                batch.append(entry)
            except asyncio.TimeoutError:
                break

        if batch:
            try:
                await _flush_batch(batch)
            except Exception:
                pass  # Don't crash the writer on a DB error; entries are lost but the app continues
            batch = []
