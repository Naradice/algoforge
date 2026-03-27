"""
StructuredLogger — writes structured log entries to the `logs` table via log_writer.

The current run IDs are propagated automatically via ContextVar so callers never
need to pass them explicitly:

    from .logging import StructuredLogger, current_strategy_run_id

    logger = StructuredLogger("strategy.handler.ml_model")

    # In executor, before running:
    token = current_strategy_run_id.set(run.id)
    try:
        await run_strategy(...)
    finally:
        current_strategy_run_id.reset(token)

    # In handler (no run_id needed):
    await logger.info("ML condition evaluated", context={"passed": True, "confidence": 0.87})
"""

import logging
from contextvars import ContextVar
from typing import Any

# These are set by each executor before entering the run loop.
current_strategy_run_id: ContextVar[int | None] = ContextVar("strategy_run_id", default=None)
current_training_run_id: ContextVar[int | None] = ContextVar("training_run_id", default=None)
current_collection_job_id: ContextVar[int | None] = ContextVar("collection_job_id", default=None)


class StructuredLogger:
    def __init__(self, source: str) -> None:
        self._source = source
        self._py = logging.getLogger(source)

    async def debug(self, message: str, context: dict[str, Any] | None = None, event_id: int | None = None) -> None:
        await self._write("DEBUG", message, context, event_id)

    async def info(self, message: str, context: dict[str, Any] | None = None, event_id: int | None = None) -> None:
        await self._write("INFO", message, context, event_id)

    async def warning(self, message: str, context: dict[str, Any] | None = None, event_id: int | None = None) -> None:
        await self._write("WARNING", message, context, event_id)

    async def error(self, message: str, context: dict[str, Any] | None = None, event_id: int | None = None) -> None:
        await self._write("ERROR", message, context, event_id)

    async def critical(self, message: str, context: dict[str, Any] | None = None, event_id: int | None = None) -> None:
        await self._write("CRITICAL", message, context, event_id)

    async def _write(self, level: str, message: str, context: dict[str, Any] | None, event_id: int | None) -> None:
        # Always emit to Python logging (console in dev)
        self._py.log(getattr(logging, level), message, extra={"context": context})

        # Fire-and-forget DB write via the batch writer
        from .log_writer import enqueue_log  # late import to avoid circular

        enqueue_log(
            level=level,
            source=self._source,
            message=message,
            context=context,
            strategy_run_id=current_strategy_run_id.get(),
            training_run_id=current_training_run_id.get(),
            collection_job_id=current_collection_job_id.get(),
            event_id=event_id,
        )
