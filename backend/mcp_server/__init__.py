"""
AlgoForge MCP Server — exposes system state as tools for AI agents.

Mount in main.py:
    from mcp_server import mcp
    app.mount("/mcp", mcp.get_asgi_app())

Or run standalone:
    python -m mcp_server

Tools are grouped by domain and registered in mcp_server/tools/.
"""

from __future__ import annotations

from fastmcp import FastMCP  # type: ignore

mcp = FastMCP(
    name="AlgoForge",
    instructions=(
        "AlgoForge is an algorithmic trading platform. "
        "Use these tools to inspect strategy performance, ML model metrics, "
        "dataset characteristics, and system logs. "
        "Always start with list_* tools to discover available resources, "
        "then drill down with detail tools."
    ),
)

# Register all tool modules (import triggers @mcp.tool() decoration)
from mcp_server.tools import logs, strategy, model, data  # noqa: E402, F401

import sqlalchemy as sa


@mcp.resource("algoforge://strategies/{strategy_id}")
async def strategy_resource(strategy_id: str) -> str:
    """Returns the strategy definition as formatted JSON."""
    import json
    from database import async_session_factory
    from strategy.models import Strategy

    async with async_session_factory() as db:
        s = (await db.execute(sa.select(Strategy).where(Strategy.id == int(strategy_id)))).scalar_one_or_none()
    if s is None:
        return json.dumps({"error": f"Strategy {strategy_id} not found"})
    return json.dumps({"id": s.id, "name": s.name, "definition": s.definition}, indent=2)


@mcp.resource("algoforge://strategies/{strategy_id}/runs/{run_id}/metrics")
async def run_metrics_resource(strategy_id: str, run_id: str) -> str:
    """Returns run metrics as formatted JSON."""
    import json
    from database import async_session_factory
    from strategy.models import RunMetric

    async with async_session_factory() as db:
        rows = (await db.execute(sa.select(RunMetric).where(RunMetric.run_id == int(run_id)))).scalars().all()
    metrics = {m.key: m.value for m in rows}
    return json.dumps({"run_id": int(run_id), "metrics": metrics}, indent=2)


@mcp.resource("algoforge://datasets/{dataset_id}/characteristics")
async def dataset_characteristics_resource(dataset_id: str) -> str:
    """Returns dataset characteristics as formatted JSON."""
    import json
    from database import async_session_factory
    from data.models import DataCharacteristics

    async with async_session_factory() as db:
        result = await db.execute(
            sa.select(DataCharacteristics).where(DataCharacteristics.dataset_id == int(dataset_id)).order_by(DataCharacteristics.computed_at.desc()).limit(1)
        )
        chars = result.scalar_one_or_none()
    if chars is None:
        return json.dumps({"error": f"No characteristics computed for dataset {dataset_id}"})
    return json.dumps({"dataset_id": int(dataset_id), "metrics": chars.metrics, "computed_at": chars.computed_at.isoformat()}, indent=2)


@mcp.resource("algoforge://dashboard")
async def dashboard_resource() -> str:
    """Platform-wide summary: active runs, recent signals, pending training jobs."""
    import json
    from database import async_session_factory
    from strategy.models import StrategyRun
    from model.models import TrainingRun

    async with async_session_factory() as db:
        active_runs = (await db.execute(
            sa.select(sa.func.count()).select_from(StrategyRun).where(StrategyRun.status.in_(["running", "pending"]))
        )).scalar_one()
        active_training = (await db.execute(
            sa.select(sa.func.count()).select_from(TrainingRun).where(TrainingRun.status.in_(["running", "pending"]))
        )).scalar_one()
        recent_runs = (await db.execute(
            sa.select(StrategyRun).order_by(StrategyRun.created_at.desc()).limit(5)
        )).scalars().all()

    return json.dumps({
        "active_strategy_runs": active_runs,
        "active_training_runs": active_training,
        "recent_strategy_runs": [
            {"id": r.id, "strategy_id": r.strategy_id, "status": r.status, "mode": r.mode}
            for r in recent_runs
        ],
    }, indent=2)
