"""

Strategy execution engine.



- Backtest mode: bar-by-bar replay via strategy.engine.backtest

- Paper/live mode: implemented in Phase 5

"""



from __future__ import annotations



from datetime import datetime, timezone



from loger import StructuredLogger, current_strategy_run_id

from webhooks.dispatcher import dispatch



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



        if "universe" in (strategy.definition or {}):

            return await _run_portfolio(run, strategy, session_factory)



        from data.models import Dataset

        ds_rec = None

        if run.dataset_id:

            result = await db.execute(select(Dataset).where(Dataset.id == run.dataset_id))

            ds_rec = result.scalar_one_or_none()



        strat_type = (strategy.definition or {}).get("strategy_type", "")

        if ds_rec is None and strat_type.startswith("analyst_"):

            sym = (strategy.definition or {}).get("symbol", "USDJPY")

            result = await db.execute(

                select(Dataset)

                .where(Dataset.name.ilike(f"%{sym}%M30%"), Dataset.artifact_path.isnot(None))

                .order_by(Dataset.id.desc())

            )

            ds_rec = result.scalars().first()



        if run.mode == "backtest" and (ds_rec is None or ds_rec.artifact_path is None):

            await db.execute(

                update(StrategyRun).where(StrategyRun.id == run_id).values(

                    status="error", message="Dataset not found or has no artifact"

                )

            )

            await dispatch(db, "run.error", {

                "run_id": run_id, "strategy_id": run.strategy_id, "mode": run.mode,

                "error": "dataset_required_for_backtest",

            })

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



    if run.mode in ("paper", "live"):

        from strategy.live_runner import run_live

        return await run_live(run_id, definition, model_cache=model_cache, mode=run.mode, risk_override=run.risk_override, strategy_id=run.strategy_id)



    if run.mode != "backtest":

        async with session_factory() as db:

            await db.execute(

                update(StrategyRun).where(StrategyRun.id == run_id).values(

                    status="error", message=f"Unknown mode: {run.mode!r}"

                )

            )

            await dispatch(db, "run.error", {

                "run_id": run_id, "strategy_id": run.strategy_id, "mode": run.mode,

                "error": "unknown_mode",

            })

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

    win_size = run.window_size or 0

    start_cap = run.starting_capital or 1.0

    _watcher = asyncio.create_task(_progress_watcher())

    try:

        raw_trades, metrics, equity_curve = await asyncio.get_running_loop().run_in_executor(

            None,

            functools.partial(

                run_backtest,

                definition,

                ds_rec.artifact_path,

                on_progress=_on_progress,

                model_cache=model_cache,

                from_ts=run.from_ts,
                to_ts=run.to_ts,
                walk_forward_ratio=wf_ratio,

                window_size=win_size,

                starting_capital=start_cap,

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

            await dispatch(db, "run.error", {

                "run_id": run_id, "strategy_id": run.strategy_id, "mode": run.mode, "error": str(e),

            })

            await db.commit()

        return {"error": str(e)}

    finally:

        _watcher.cancel()

        with contextlib.suppress(asyncio.CancelledError):

            await _watcher



    # ── Persist trades + metrics ──────────────────────────────────────────────

    try:

        async with session_factory() as db:

            for t in raw_trades:

                db.add(Trade(

                    run_id=run_id,

                    symbol=t["symbol"],

                    direction=t["direction"],

                    entry_price=t["entry_price"],

                    exit_price=t["exit_price"],

                    volume=t["volume"],

                    sl_price=t.get("sl_price"),

                    tp_price=t.get("tp_price"),

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

            await dispatch(db, "run.completed", {

                "run_id": run_id, "strategy_id": run.strategy_id, "mode": run.mode,

                "trades": len(raw_trades), "metrics": metrics,

            })

            await db.commit()

    except Exception as e:

        await logger.error('Failed to persist backtest results', context={'run_id': run_id, 'error': str(e)})

        async with session_factory() as db:

            await db.execute(

                update(StrategyRun).where(StrategyRun.id == run_id).values(

                    status='error', message=f'Persist error: {e}', ended_at=datetime.now(timezone.utc)

                )

            )

            await db.commit()

        return {'error': str(e)}



    await logger.info(

        "Backtest completed",

        context={"run_id": run_id, "trades": len(raw_trades), "metrics": metrics},

    )

    return {"trades": len(raw_trades), "metrics": metrics}





async def _run_portfolio(run, strategy, session_factory) -> dict:

    """Multi-symbol (portfolio) strategies: backtest and paper (re-computed target book) in one code path."""

    import asyncio

    import functools



    from sqlalchemy import update

    from strategy.models import RunMetric, StrategyEvent, StrategyRun

    from strategy.portfolio.engine import run_portfolio_backtest



    definition = dict(strategy.definition)

    if run.risk_override:

        definition["risk"] = {**definition.get("risk", {}), **run.risk_override}

    async with session_factory() as db:

        await db.execute(update(StrategyRun).where(StrategyRun.id == run.id).values(

            status="running", started_at=datetime.now(timezone.utc), progress_pct=0.0))

        await db.commit()

    try:

        res = await asyncio.get_running_loop().run_in_executor(

            None, functools.partial(run_portfolio_backtest, definition))

    except Exception as exc:

        async with session_factory() as db:

            await db.execute(update(StrategyRun).where(StrategyRun.id == run.id).values(

                status="error", message=str(exc), ended_at=datetime.now(timezone.utc)))

            await dispatch(db, "run.error", {"run_id": run.id, "strategy_id": run.strategy_id,

                                             "mode": run.mode, "error": str(exc)})

            await db.commit()

        return {"error": str(exc)}



    eq = res["equity"].resample("W").last().dropna()

    cum = res["returns"].cumsum()

    dd = (cum - cum.cummax()).resample("W").last().reindex(eq.index).fillna(0.0)

    curve = [{"timestamp": t.isoformat(), "equity": round(float(v), 6), "drawdown": round(float(dd.loc[t]), 6),

              "phase": "oos"} for t, v in eq.items()]

    m = res["metrics"]

    flat = {k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and v is not None}

    async with session_factory() as db:

        for k, v in flat.items():

            db.add(RunMetric(run_id=run.id, key=k, value=v))

        for name, value in res["stress"].items():

            if value is not None:

                db.add(RunMetric(run_id=run.id, key="stress: " + name, value=float(value)))

        db.add(StrategyEvent(run_id=run.id, strategy_id=run.strategy_id, event_type="portfolio.targets",

                             payload=res["targets"], occurred_at=datetime.now(timezone.utc)))

        sr = flat.get("sharpe")

        await db.execute(update(StrategyRun).where(StrategyRun.id == run.id).values(

            status="completed", progress_pct=100.0, ended_at=datetime.now(timezone.utc),

            message=f"portfolio {run.mode}: Sharpe {sr:.2f}, DD {flat.get('max_drawdown', 0):.3f}, "

                    f"asof {res['targets'].get('asof')}" if sr is not None else "portfolio run: no data",

            equity_curve=curve))

        await dispatch(db, "run.completed", {"run_id": run.id, "strategy_id": run.strategy_id, "mode": run.mode,

                                             "metrics": flat, "targets": res["targets"]})

        await db.commit()

    return {"metrics": flat, "targets": res["targets"]}





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

