"""
Shared arq pool for enqueueing background jobs from the FastAPI app.

Initialised once at startup via main.py lifespan. Use get_arq_pool() in services.

    from .arq_pool import get_arq_pool
    pool = await get_arq_pool()
    await pool.enqueue_job("run_collection_job", job_id)
"""

from __future__ import annotations

import os

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

_pool: ArqRedis | None = None

_NO_REDIS = os.getenv("ALGOFORGE_NO_REDIS", "").lower() in ("1", "true")


async def init_arq_pool() -> None:
    global _pool
    if _NO_REDIS:
        return
    _pool = await create_pool(RedisSettings.from_dsn(os.getenv("REDIS_URL", "redis://localhost:6379/0")))


async def get_arq_pool() -> ArqRedis | None:
    return _pool


async def enqueue(job_name: str, *args, **kwargs) -> None:
    """Fire-and-forget enqueue. No-ops when Redis is disabled (ALGOFORGE_NO_REDIS=1)."""
    if _pool is None:
        return
    await _pool.enqueue_job(job_name, *args, **kwargs)
