"""
Strategy execution engine.

- Backtest mode: bar-by-bar replay via strategy.engine.backtest
- Paper/live mode: implemented in Phase 5
"""

from __future__ import annotations

from datetime import datetime, timezone

from loger import StructuredLogger, current_strategy_run_id

logger = StructuredLogger("strategy.executor")


async def execute_strategy_run(run_id: int, *, session_factory=None) -> dict:
    """Dispatches to backtest or live runner.

    session_factory: optional async_sessionmaker. When called from a Celery
    worker, pass a NullPool-based factory to avoid asyncio event-loop conflicts.
    Falls back to the module-level factory (suitable for FastAPI context).
    """
    token = current_strategy_run_id.set(run_id)
    try:
        return await _run(run_id, session_factory=session_factory)
    finally:
        current_strategy_run_id.reset(token)


async def _run(run_id: int, *, session_factory=None) -> dict:
    import asyncio
    if session_factory is None:
        from database import async_session_factory as session_factory
    from strategy.models import StrategyRun, Strategy, Trade, RunMetric
    from strategy.engine.backtest import run_backtest
    from sqlalchemy import select, update

    async with session_factory() as db:
        result = await db.execute(select(StrategyRun).where(StrategyRun.id == run_id))
        run = result.scalar_one_or_none()
        if run is None:
            await logger.error("StrategyRun not found", context={"run_id": run_id})
            return {"error": "run_not_found"}

        result = await db.execute(select(Strategy).where(Strategy.id == run.strategy_id))
        strategy = result.scalar_one_or_none()
        if strategy is None:
            return {"error": "strategy_not_found"}

        from data.models import Dataset
        ds_rec = None
        if run.dataset_id:
            result = await db.execute(select(Dataset).where(Dataset.id == run.dataset_id))
            ds_rec = result.scalar_one_or_none()

        if run.mode == "backtest" and (ds_rec is None or ds_rec.artifact_path is None):
            await db.execute(
                update(StrategyRun).where(StrategyRun.id == run_id).values(
                    status="error", message="Dataset not found or has no artifact"
                )
            )
            await db.commit()
            return {"error": "dataset_required_for_backtest"}

        await db.execute(
            update(StrategyRun).where(StrategyRun.id == run_id).values(
                status="running", started_at=datetime.now(timezone.utc), progress_pct=0.0
            )
        )
        await db.commit()

    # ── Merge risk_override into definition ──────────────────────────────────────
    definition = dict(strategy.definition)
    if run.risk_override:
        definition = {**definition, "risk": {**definition.get("risk", {}), **run.risk_override}}

    # ── Pre-load ML model metadata for any ml_signal conditions ─────────────────
    model_cache = await _load_model_cache(definition, session_factory)

    await logger.info("Strategy run started", context={"run_id": run_id, "mode": run.mode})

    if run.mode == "paper":
        from strategy.live_runner import run_paper
        return await run_paper(run_id, definition, model_cache)

    if run.mode == "live":
        async with session_factory() as db:
            await db.execute(
                update(StrategyRun).where(StrategyRun.id == run_id).values(
                    status="error", message="Live broker execution not yet implemented"
                )
            )
            await db.commit()
        return {"error": "live_not_implemented"}

    if run.mode != "backtest":
        async with session_factory() as db:
            await db.execute(
                update(StrategyRun).where(StrategyRun.id == run_id).values(
                    status="error", message=f"Unknown mode: {run.mode!r}"
                )
            )
            await db.commit()
        return {"error": "unknown_mode"}

    # ── Backtest ──────────────────────────────────────────────────────────────
    # _progress_pct is written by the backtest thread and read by the async
    # watcher task. A one-element list is used so the reference is stable
    # inside the closures; CPython's GIL makes single-slot float writes atomic.
    _progress_pct: list[float] = [0.0]

    def _on_progress(pct: float) -> None:
        _progress_pct[0] = pct

    async def _progress_watcher() -> None:
        """Polls the shared progress value every 5 s and flushes it to the DB."""
        last_written = -1.0
        while True:
            await asyncio.sleep(5)
            current = _progress_pct[0]
            if current - last_written >= 5.0:
                last_written = current
                try:
                    async with session_factory() as db:
                        await db.execute(
                            update(StrategyRun).where(StrategyRun.id == run_id).values(progress_pct=current)
                        )
                        await db.commit()
                except Exception:
                    pass  # non-fatal; final commit below will capture the end state

    import contextlib
    import functools
    wf_ratio = run.walk_forward_ratio or 0.0
    _watcher = asyncio.create_task(_progress_watcher())
    try:
        raw_trades, metrics, equity_curve = await asyncio.get_event_loop().run_in_executor(
            None,
            functools.partial(
                run_backtest,
                definition,
                ds_rec.artifact_path,
                on_progress=_on_progress,
                model_cache=model_cache,
                walk_forward_ratio=wf_ratio,
            ),
        )
    except Exception as e:
        await logger.error("Backtest failed", context={"run_id": run_id, "error": str(e)})
        async with session_factory() as db:
            await db.execute(
                update(StrategyRun).where(StrategyRun.id == run_id).values(
                    status="error", message=str(e), ended_at=datetime.now(timezone.utc)
                )
            )
            await db.commit()
        return {"error": str(e)}
    finally:
        _watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _watcher

    # ── Persist trades + metrics ──────────────────────────────────────────────
    async with session_factory() as db:
        for t in raw_trades:
            db.add(Trade(
                run_id=run_id,
                symbol=t["symbol"],
                direction=t["direction"],
                entry_price=t["entry_price"],
                exit_price=t["exit_price"],
                volume=t["volume"],
                sl_price=t["sl_price"],
                tp_price=t["tp_price"],
                profit=t["profit"],
                opened_at=t["opened_at"],
                closed_at=t["closed_at"],
                exit_reason=t.get("exit_reason"),
                phase=t.get("phase"),
                mae=t.get("mae"),
                mfe=t.get("mfe"),
            ))

        for key, value in metrics.items():
            db.add(RunMetric(run_id=run_id, key=key, value=float(value)))

        await db.execute(
            update(StrategyRun).where(StrategyRun.id == run_id).values(
                status="completed",
                progress_pct=100.0,
                ended_at=datetime.now(timezone.utc),
                message=f"{metrics.get('total_trades', 0)} trades, PnL {metrics.get('total_pnl', 0):.4f}",
                equity_curve=equity_curve,
            )
        )
        await db.commit()

    await logger.info(
        "Backtest completed",
        context={"run_id": run_id, "trades": len(raw_trades), "metrics": metrics},
    )
    return {"trades": len(raw_trades), "metrics": metrics}


async def _load_model_cache(definition: dict, session_factory) -> dict:
    """
    Scan all condition blocks in a strategy definition for ml_signal conditions,
    load the corresponding MLModel records from DB, and return a model_cache dict
    suitable for strategy.engine.ml_condition.evaluate_ml_condition().
    """
    model_ids: set[int] = set()
    # Old format: entry/exit at top level
    for block_key in ("entry", "exit"):
        block = definition.get(block_key, {})
        for cond in block.get("conditions", []):
            if cond.get("type") == "ml_signal" and "model_id" in cond:
                model_ids.add(int(cond["model_id"]))
    # New format: long/short → entry/exit
    for side in ("long", "short"):
        side_def = definition.get(side, {})
        for block_key in ("entry", "exit"):
            block = side_def.get(block_key, {})
            for cond in block.get("conditions", []):
                if cond.get("type") == "ml_signal" and "model_id" in cond:
                    model_ids.add(int(cond["model_id"]))

    if not model_ids:
        return {}

    from model.models import MLModel, TrainingRun
    from sqlalchemy import select

    cache: dict[int, dict] = {}
    async with session_factory() as db:
        for mid in model_ids:
            result = await db.execute(select(MLModel).where(MLModel.id == mid))
            model_rec = result.scalar_one_or_none()
            if model_rec is None or model_rec.artifact_path is None:
                logger.warning(f"ml_signal: model {mid} not found or not deployed — skipping")
                continue

            # Find the training run whose artifact matches the deployed artifact
            result = await db.execute(
                select(TrainingRun).where(
                    TrainingRun.model_id == mid,
                    TrainingRun.artifact_path == model_rec.artifact_path,
                )
            )
            run_rec = result.scalar_one_or_none()
            hyperparams = run_rec.hyperparams if run_rec else {}

            cache[mid] = {
                "architecture": model_rec.architecture,
                "config": model_rec.config,
                "artifact_path": model_rec.artifact_path,
                "hyperparams": hyperparams,
            }

    return cache
