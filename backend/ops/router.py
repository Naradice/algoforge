"""Operational status endpoints (/ops) -- see ops/worker_registry.py."""
from __future__ import annotations

from fastapi import APIRouter

from schemas import DataResponse

ops_router = APIRouter(prefix="/ops", tags=["ops"])


@ops_router.get("/queues")
async def queue_status():
    """Queues (pending messages, registered/busy workers) and workers (start time, code revision,
    stale_code vs repository HEAD, current task, host memory)."""
    import asyncio
    from ops.worker_registry import get_queue_status
    return DataResponse(data=await asyncio.to_thread(get_queue_status))
