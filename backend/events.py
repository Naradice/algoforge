"""
Simple in-process event bus for SSE streaming.
Channels are keyed by string (e.g. "run:5", "training:12").
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventBus:
    _subscribers: dict[str, list[asyncio.Queue]] = field(default_factory=dict)

    async def publish(self, channel: str, event: Any) -> None:
        for q in self._subscribers.get(channel, []):
            await q.put(event)

    @asynccontextmanager
    async def subscribe(self, channel: str):
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(channel, []).append(q)
        try:
            yield q
        finally:
            self._subscribers[channel].remove(q)
            if not self._subscribers[channel]:
                del self._subscribers[channel]


event_bus = EventBus()
