"""MCP tools — structured log querying."""

from __future__ import annotations

import sqlalchemy as sa

from mcp_server import mcp


_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}


def _level_filter(min_level: str) -> list[str]:
    min_order = _LEVEL_ORDER.get(min_level.upper(), 1)
    return [k for k, v in _LEVEL_ORDER.items() if v >= min_order]


@mcp.tool()
async def get_run_logs(
    run_id: int,
    run_type: str = "strategy",
    level: str = "INFO",
    limit: int = 50,
) -> list[dict]:
    """
    Get structured logs for a specific run.

    Args:
        run_id:   ID of the run (strategy_run, training_run, or collection_job).
        run_type: 'strategy' | 'training' | 'collection'
        level:    Minimum log level: DEBUG | INFO | WARNING | ERROR | CRITICAL
        limit:    Maximum number of entries to return (newest first, max 200).
    """
    from database import async_session_factory
    from log_models import Log

    limit = min(limit, 200)
    q = sa.select(Log).where(Log.level.in_(_level_filter(level)))

    if run_type == "strategy":
        q = q.where(Log.strategy_run_id == run_id)
    elif run_type == "training":
        q = q.where(Log.training_run_id == run_id)
    elif run_type == "collection":
        q = q.where(Log.collection_job_id == run_id)
    else:
        return [{"error": f"Unknown run_type: {run_type!r}. Use 'strategy', 'training', or 'collection'."}]

    q = q.order_by(Log.created_at.desc()).limit(limit)

    async with async_session_factory() as db:
        rows = (await db.execute(q)).scalars().all()

    return [
        {
            "id": r.id,
            "ts": r.created_at.isoformat(),
            "level": r.level,
            "source": r.source,
            "message": r.message,
            "context": r.context,
        }
        for r in rows
    ]


@mcp.tool()
async def search_logs(
    query: str,
    level: str = "WARNING",
    limit: int = 50,
) -> list[dict]:
    """
    Full-text search in log messages across all runs.

    Args:
        query: Substring to search for in log messages (case-insensitive).
        level: Minimum log level to include (default: WARNING).
        limit: Max entries to return (max 200).
    """
    from database import async_session_factory
    from log_models import Log

    limit = min(limit, 200)
    q = (
        sa.select(Log)
        .where(Log.level.in_(_level_filter(level)))
        .where(Log.message.ilike(f"%{query}%"))
        .order_by(Log.created_at.desc())
        .limit(limit)
    )

    async with async_session_factory() as db:
        rows = (await db.execute(q)).scalars().all()

    return [
        {
            "id": r.id,
            "ts": r.created_at.isoformat(),
            "level": r.level,
            "source": r.source,
            "message": r.message,
            "strategy_run_id": r.strategy_run_id,
            "training_run_id": r.training_run_id,
        }
        for r in rows
    ]


@mcp.tool()
async def get_log_summary(
    run_id: int | None = None,
    run_type: str = "strategy",
) -> dict:
    """
    Get a summary of log counts grouped by level and source for a run,
    plus the first error message if any.

    Args:
        run_id:   Optional run ID to scope the summary. If None, summarises all logs.
        run_type: 'strategy' | 'training' | 'collection'
    """
    from database import async_session_factory
    from log_models import Log

    q = sa.select(Log)
    if run_id is not None:
        if run_type == "strategy":
            q = q.where(Log.strategy_run_id == run_id)
        elif run_type == "training":
            q = q.where(Log.training_run_id == run_id)
        elif run_type == "collection":
            q = q.where(Log.collection_job_id == run_id)

    async with async_session_factory() as db:
        rows = (await db.execute(q)).scalars().all()

    counts_by_level: dict[str, int] = {}
    counts_by_source: dict[str, int] = {}
    first_error: dict | None = None

    for r in rows:
        counts_by_level[r.level] = counts_by_level.get(r.level, 0) + 1
        counts_by_source[r.source] = counts_by_source.get(r.source, 0) + 1
        if r.level in ("ERROR", "CRITICAL") and first_error is None:
            first_error = {
                "ts": r.created_at.isoformat(),
                "source": r.source,
                "message": r.message,
                "context": r.context,
            }

    return {
        "total": len(rows),
        "counts_by_level": counts_by_level,
        "counts_by_source": counts_by_source,
        "first_error": first_error,
    }
