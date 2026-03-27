"""
Internal event bus.

In production: Redis Streams (one stream per event type).
In development (ALGOFORGE_NO_REDIS=1): in-process asyncio queues — no Redis needed.

Usage:
    from .events import event_bus

    # Publish
    await event_bus.publish("strategy.tick", {"run_id": 42, "symbol": "EURUSD", ...})

    # Subscribe (typically in a background task started at app startup)
    async for event in event_bus.subscribe("strategy.tick"):
        await handle_tick(event)
"""

import asyncio
import json
import os
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Any


class InProcessEventBus:
    """Asyncio queue-based bus for dev/test. No external dependencies."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        message = {"type": event_type, "data": payload}
        for queue in self._subscribers[event_type]:
            await queue.put(message)

    async def subscribe(self, event_type: str) -> AsyncGenerator[dict[str, Any], None]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers[event_type].append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers[event_type].remove(queue)


class RedisEventBus:
    """Redis Streams-backed event bus for production."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis: Any = None  # redis.asyncio.Redis — imported lazily

    async def _get_redis(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis  # type: ignore

            self._redis = await aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        r = await self._get_redis()
        stream_key = f"algoforge:{event_type}"
        await r.xadd(stream_key, {"data": json.dumps(payload)}, maxlen=10_000, approximate=True)

    async def subscribe(self, event_type: str) -> AsyncGenerator[dict[str, Any], None]:
        r = await self._get_redis()
        stream_key = f"algoforge:{event_type}"
        last_id = "$"
        while True:
            entries = await r.xread({stream_key: last_id}, block=5000, count=100)
            for _stream, messages in entries:
                for msg_id, fields in messages:
                    last_id = msg_id
                    yield {"type": event_type, "data": json.loads(fields["data"])}


_NO_REDIS = os.getenv("ALGOFORGE_NO_REDIS", "").lower() in ("1", "true")
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

event_bus: InProcessEventBus | RedisEventBus = (
    InProcessEventBus() if _NO_REDIS else RedisEventBus(_REDIS_URL)
)
