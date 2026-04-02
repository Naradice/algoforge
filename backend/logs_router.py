"""Logs — HTTP endpoints for querying structured logs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from log_models import Log
from schemas import DataResponse

router = APIRouter(prefix="/logs", tags=["logs"])


class LogEntryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    level: str
    source: str
    message: str
    context: dict[str, Any] | None
    strategy_run_id: int | None
    training_run_id: int | None
    collection_job_id: int | None
    event_id: int | None


class LogSummary(BaseModel):
    counts_by_level: dict[str, int]
    counts_by_source: dict[str, int]
    first_error: LogEntryRead | None


_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}


@router.get("", response_model=DataResponse[list[LogEntryRead]])
async def get_logs(
    strategy_run_id: int | None = None,
    training_run_id: int | None = None,
    collection_job_id: int | None = None,
    event_id: int | None = None,
    level: str = Query("INFO", description="Minimum level: DEBUG | INFO | WARNING | ERROR | CRITICAL"),
    source: str | None = None,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
    q: str | None = None,
    limit: int = Query(200, le=1000),
    cursor: int | None = None,  # log id for cursor-based pagination
    db: AsyncSession = Depends(get_db),
):
    min_order = _LEVEL_ORDER.get(level.upper(), 1)
    level_filter = [k for k, v in _LEVEL_ORDER.items() if v >= min_order]

    query = sa.select(Log).where(Log.level.in_(level_filter))

    if strategy_run_id is not None:
        query = query.where(Log.strategy_run_id == strategy_run_id)
    if training_run_id is not None:
        query = query.where(Log.training_run_id == training_run_id)
    if collection_job_id is not None:
        query = query.where(Log.collection_job_id == collection_job_id)
    if event_id is not None:
        query = query.where(Log.event_id == event_id)
    if source:
        query = query.where(Log.source.like(f"{source}%"))
    if from_ts:
        query = query.where(Log.created_at >= from_ts)
    if to_ts:
        query = query.where(Log.created_at <= to_ts)
    if q:
        query = query.where(Log.message.ilike(f"%{q}%"))
    if cursor:
        query = query.where(Log.id > cursor)

    query = query.order_by(Log.created_at.desc()).limit(limit)
    result = await db.execute(query)
    return DataResponse(data=list(result.scalars().all()))


@router.get("/summary", response_model=DataResponse[LogSummary])
async def get_log_summary(
    strategy_run_id: int | None = None,
    training_run_id: int | None = None,
    collection_job_id: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    def _apply_filters(q):
        if strategy_run_id:
            q = q.where(Log.strategy_run_id == strategy_run_id)
        if training_run_id:
            q = q.where(Log.training_run_id == training_run_id)
        if collection_job_id:
            q = q.where(Log.collection_job_id == collection_job_id)
        return q

    # Counts by level — single GROUP BY query, no full scan
    level_rows = (await db.execute(
        _apply_filters(sa.select(Log.level, sa.func.count().label("n")).group_by(Log.level))
    )).all()
    counts_by_level = {row.level: row.n for row in level_rows}

    # Counts by source — single GROUP BY query
    source_rows = (await db.execute(
        _apply_filters(sa.select(Log.source, sa.func.count().label("n")).group_by(Log.source))
    )).all()
    counts_by_source = {row.source: row.n for row in source_rows}

    # First error — LIMIT 1, no table scan
    first_error_row = (await db.execute(
        _apply_filters(
            sa.select(Log)
            .where(Log.level.in_(("ERROR", "CRITICAL")))
            .order_by(Log.id.asc())
            .limit(1)
        )
    )).scalar_one_or_none()
    first_error = LogEntryRead.model_validate(first_error_row) if first_error_row else None

    return DataResponse(data=LogSummary(counts_by_level=counts_by_level, counts_by_source=counts_by_source, first_error=first_error))
