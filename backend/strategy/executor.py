"""
Strategy execution engine (stub — implemented in Phase 3 for backtest, Phase 5 for live).

Responsibilities:
- Set current_strategy_run_id ContextVar so all log calls include the run ID
- Drive the event loop: emit events, evaluate conditions, apply logic, execute trades
- Update strategy_runs.status and strategy_runs.progress_pct in DB
- Write StrategyEvent rows and Trade rows to DB
- Publish results to the event bus (→ WebSocket pusher, webhook dispatcher)
"""

from __future__ import annotations

import asyncio

from ..logging import StructuredLogger, current_strategy_run_id

logger = StructuredLogger("strategy.executor")


async def execute_strategy_run(run_id: int) -> None:
    """arq job entry point. Called by the worker when a run is enqueued."""
    token = current_strategy_run_id.set(run_id)
    try:
        await logger.info("Strategy run started", context={"run_id": run_id})
        # TODO Phase 3: implement backtest execution
        # TODO Phase 5: implement live/paper execution
        raise NotImplementedError("Strategy execution not yet implemented — Phase 3")
    except NotImplementedError:
        await logger.warning("Strategy execution not yet implemented", context={"run_id": run_id})
    except Exception as e:
        import traceback
        await logger.error("Strategy run failed", context={"run_id": run_id, "exc": str(e), "traceback": traceback.format_exc()})
        raise
    finally:
        current_strategy_run_id.reset(token)
