"""
Condition handler registry.

A handler evaluates whether a condition is met given an event payload.
Register new handlers here after implementing them.

Usage:
    from .handlers import HANDLER_REGISTRY
    handler_cls = HANDLER_REGISTRY["macd_renko"]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventData:
    event_id: int
    event_type: str
    payload: dict[str, Any]
    run_id: int
    strategy_id: int


@dataclass
class ConditionResult:
    result: bool
    details: dict[str, Any] = field(default_factory=dict)


class ConditionHandler(ABC):
    name: str = ""
    params_schema: dict = {}

    @abstractmethod
    async def evaluate(self, event_data: EventData, params: dict) -> ConditionResult:
        ...


# Registry — populated as handlers are implemented in later phases
HANDLER_REGISTRY: dict[str, type[ConditionHandler]] = {}
