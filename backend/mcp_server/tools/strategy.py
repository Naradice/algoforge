"""MCP tools — strategy performance and trade inspection."""

from __future__ import annotations

import sqlalchemy as sa

from mcp_server import mcp


@mcp.tool()
async def list_strategies() -> list[dict]:
    """
    List all strategies in the system with their current status and last run summary.
    Use this first to discover available strategies before drilling down.
    """
    from database import async_session_factory
    from strategy.models import Strategy, StrategyRun

    async with async_session_factory() as db:
        strategies = (
            await db.execute(
                sa.select(Strategy).order_by(Strategy.created_at.desc())
            )
        ).scalars().all()

        result = []
        for s in strategies:
            # Get most recent run
            run_row = (
                await db.execute(
                    sa.select(StrategyRun)
                    .where(StrategyRun.strategy_id == s.id)
                    .order_by(StrategyRun.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

            result.append({
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "status": s.status,
                "created_at": s.created_at.isoformat(),
                "last_run": {
                    "id": run_row.id,
                    "mode": run_row.mode,
                    "status": run_row.status,
                    "message": run_row.message,
                    "started_at": run_row.started_at.isoformat() if run_row.started_at else None,
                } if run_row else None,
            })

    return result


@mcp.tool()
async def get_strategy_definition(strategy_id: int) -> dict:
    """
    Get the full definition of a strategy (indicators, entry/exit conditions, risk params).

    Args:
        strategy_id: ID of the strategy.
    """
    from database import async_session_factory
    from strategy.models import Strategy

    async with async_session_factory() as db:
        s = (
            await db.execute(sa.select(Strategy).where(Strategy.id == strategy_id))
        ).scalar_one_or_none()

    if s is None:
        return {"error": f"Strategy {strategy_id} not found"}

    return {
        "id": s.id,
        "name": s.name,
        "description": s.description,
        "definition": s.definition,
        "status": s.status,
    }


@mcp.tool()
async def get_strategy_runs(strategy_id: int, limit: int = 10) -> list[dict]:
    """
    Get recent runs for a strategy, ordered newest first.

    Args:
        strategy_id: ID of the strategy.
        limit:       Max number of runs to return (max 50).
    """
    from database import async_session_factory
    from strategy.models import StrategyRun

    limit = min(limit, 50)

    async with async_session_factory() as db:
        rows = (
            await db.execute(
                sa.select(StrategyRun)
                .where(StrategyRun.strategy_id == strategy_id)
                .order_by(StrategyRun.created_at.desc())
                .limit(limit)
            )
        ).scalars().all()

    return [
        {
            "id": r.id,
            "mode": r.mode,
            "status": r.status,
            "progress_pct": r.progress_pct,
            "message": r.message,
            "dataset_id": r.dataset_id,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "ended_at": r.ended_at.isoformat() if r.ended_at else None,
            "duration_s": (
                (r.ended_at - r.started_at).total_seconds()
                if r.started_at and r.ended_at else None
            ),
        }
        for r in rows
    ]


@mcp.tool()
async def get_run_metrics(run_id: int) -> dict:
    """
    Get computed performance metrics for a strategy run.
    Returns win_rate, total_pnl, sharpe_ratio, max_drawdown, profit_factor, etc.

    Args:
        run_id: ID of the strategy run.
    """
    from database import async_session_factory
    from strategy.models import RunMetric, StrategyRun

    async with async_session_factory() as db:
        run = (
            await db.execute(sa.select(StrategyRun).where(StrategyRun.id == run_id))
        ).scalar_one_or_none()

        if run is None:
            return {"error": f"Run {run_id} not found"}

        metrics_rows = (
            await db.execute(sa.select(RunMetric).where(RunMetric.run_id == run_id))
        ).scalars().all()

    metrics = {m.key: m.value for m in metrics_rows}
    return {
        "run_id": run_id,
        "mode": run.mode,
        "status": run.status,
        "metrics": metrics,
        "interpretation": _interpret_metrics(metrics),
    }


@mcp.tool()
async def get_run_trades(run_id: int, limit: int = 20) -> list[dict]:
    """
    Get individual trades for a strategy run.

    Args:
        run_id: ID of the strategy run.
        limit:  Max trades to return, newest first (max 100).
    """
    from database import async_session_factory
    from strategy.models import Trade

    limit = min(limit, 100)

    async with async_session_factory() as db:
        rows = (
            await db.execute(
                sa.select(Trade)
                .where(Trade.run_id == run_id)
                .order_by(Trade.opened_at.desc())
                .limit(limit)
            )
        ).scalars().all()

    return [
        {
            "symbol": t.symbol,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "profit_pct": round(t.profit * 100, 3) if t.profit is not None else None,
            "sl_price": t.sl_price,
            "tp_price": t.tp_price,
            "opened_at": t.opened_at.isoformat(),
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
            "duration_h": (
                round((t.closed_at - t.opened_at).total_seconds() / 3600, 1)
                if t.closed_at and t.opened_at else None
            ),
        }
        for t in rows
    ]


def _interpret_metrics(m: dict) -> str:
    """Return a plain-English summary of the key metrics."""
    if not m:
        return "No metrics available yet."
    parts = []
    if "win_rate" in m:
        wr = m["win_rate"] * 100
        parts.append(f"Win rate {wr:.1f}% ({'above' if wr > 50 else 'below'} 50%)")
    if "total_pnl" in m:
        pnl = m["total_pnl"] * 100
        parts.append(f"Total PnL {pnl:+.2f}%")
    if "sharpe_ratio" in m:
        sr = m["sharpe_ratio"]
        quality = "excellent" if sr > 2 else "good" if sr > 1 else "poor" if sr < 0 else "fair"
        parts.append(f"Sharpe {sr:.2f} ({quality})")
    if "max_drawdown" in m:
        dd = m["max_drawdown"] * 100
        parts.append(f"Max drawdown {dd:.2f}%")
    if "total_trades" in m:
        parts.append(f"{int(m['total_trades'])} trades")
    return "; ".join(parts) if parts else "Metrics available but not interpretable."


@mcp.tool()
async def create_strategy(name: str, definition: dict, description: str = "") -> dict:
    """
    Create a new trading strategy.

    Args:
        name:        Human-readable strategy name.
        definition:  Strategy definition dict (symbol, indicators, entry, exit, risk).
        description: Optional notes.
    """
    from database import async_session_factory
    from strategy.service import strategy_service
    from strategy.models import StrategyCreate

    async with async_session_factory() as db:
        body = StrategyCreate(name=name, description=description, definition=definition)
        s = await strategy_service.create_strategy(db, body)
        return {"id": s.id, "name": s.name, "status": s.status}


@mcp.tool()
async def update_strategy(strategy_id: int, definition: dict) -> dict:
    """
    Update a strategy's definition.

    Args:
        strategy_id: ID of the strategy to update.
        definition:  New strategy definition dict.
    """
    from database import async_session_factory
    from strategy.service import strategy_service
    from strategy.models import StrategyUpdate

    async with async_session_factory() as db:
        body = StrategyUpdate(definition=definition)
        s = await strategy_service.update_strategy(db, strategy_id, body)
        return {"id": s.id, "name": s.name, "status": s.status}


@mcp.tool()
async def start_strategy_run(
    strategy_id: int,
    mode: str,
    dataset_id: int | None = None,
    broker_client: str | None = None,
    risk_override: dict | None = None,
    walk_forward_ratio: float | None = None,
    window_size: int | None = None,
    starting_capital: float | None = None,
    from_ts: str | None = None,
    to_ts: str | None = None,
) -> dict:
    """
    Start a strategy run (backtest, paper, or live).
    For backtest mode, dataset_id is required.
    For paper/live mode, broker_client is required.

    Args:
        strategy_id:        Strategy to run.
        mode:               "backtest" | "paper" | "live"
        dataset_id:         Dataset ID for backtest mode.
        broker_client:      Broker client name for paper/live mode.
        risk_override:      Override risk params for this run, e.g.
                            {"slippage_pct": 0.00004, "commission_pct": 0,
                             "trailing_atr_multiplier": 5}.
                            Only the keys provided are overridden; the rest
                            come from the strategy definition.
        walk_forward_ratio: 0.0–0.9 → fraction used as in-sample. 0 = disabled.
        window_size:        Rolling indicator window in bars. 0 = full history.
                            Set to 100–250 for path-dependent indicators (Renko).
        starting_capital:   Scale PnL to this capital (e.g. 150 for USDJPY).
                            1.0 = normalised (default).
        from_ts:            ISO datetime to start from (backtest only).
        to_ts:              ISO datetime to end at (backtest only).
    """
    from database import async_session_factory
    from strategy.service import strategy_service
    from strategy.models import StrategyRunCreate
    from celery_app import enqueue

    async with async_session_factory() as db:
        body = StrategyRunCreate(
            mode=mode, dataset_id=dataset_id,
            broker_client=broker_client,
            risk_override=risk_override,
            walk_forward_ratio=walk_forward_ratio,
            window_size=window_size,
            starting_capital=starting_capital,
        )
        run = await strategy_service.create_run(db, strategy_id, body)
        await enqueue("execute_strategy_run", run.id)
        return {"run_id": run.id, "strategy_id": strategy_id, "status": run.status}


@mcp.tool()
async def get_run_status(strategy_id: int, run_id: int) -> dict:
    """
    Poll a strategy run for its current status and progress.
    Returns status ("pending"|"running"|"completed"|"error"), progress_pct, message.
    Call this repeatedly until status is "completed" or "error".

    Args:
        strategy_id: Strategy ID.
        run_id:      Run ID returned by start_strategy_run.
    """
    from database import async_session_factory
    from strategy.service import strategy_service

    async with async_session_factory() as db:
        run = await strategy_service.get_run(db, strategy_id, run_id)
    return {
        "status": run.status,
        "progress_pct": run.progress_pct,
        "message": run.message,
    }


@mcp.tool()
async def compare_runs(strategy_id: int, run_ids: list[int]) -> dict:
    """
    Compare metrics across multiple strategy runs side-by-side.

    Args:
        strategy_id: Strategy ID.
        run_ids:     List of run IDs to compare.
    """
    from database import async_session_factory
    from strategy.service import strategy_service

    async with async_session_factory() as db:
        return await strategy_service.compare_runs(db, strategy_id, run_ids)


@mcp.tool()
async def stop_strategy_run(strategy_id: int, run_id: int) -> dict:
    """
    Stop an active strategy run.

    Args:
        strategy_id: Strategy ID.
        run_id:      Run ID to stop.
    """
    from database import async_session_factory
    from strategy.service import strategy_service

    async with async_session_factory() as db:
        run = await strategy_service.stop_run(db, strategy_id, run_id)
    return {"run_id": run.id, "status": run.status}


@mcp.tool()
async def send_strategy_chat(strategy_id: int, run_id: int, message: str) -> dict:
    """
    Send a chat message to a strategy run (for AI-driven analysis).

    Args:
        strategy_id: Strategy ID.
        run_id:      Run ID.
        message:     Message text to send.
    """
    from database import async_session_factory
    from strategy.service import strategy_service
    from strategy.models import ChatMessageCreate

    async with async_session_factory() as db:
        msg = await strategy_service.send_chat_message(db, strategy_id, run_id, ChatMessageCreate(message=message))
    return {"id": msg.id, "role": msg.role, "message": msg.message}


@mcp.tool()
async def run_parameter_sweep(
    strategy_id: int,
    dataset_id: int,
    variants: list[dict],
    base_risk: dict | None = None,
    starting_capital: float = 1.0,
) -> dict:
    """
    Launch multiple backtest variants in parallel and return a ranked comparison
    once all complete. Use this for systematic strategy investigation.

    Args:
        strategy_id:      Strategy to sweep.
        dataset_id:       Dataset to run all variants against.
        variants:         List of variant configs. Each entry is a dict with:
                          - "label": short name for this variant (required)
                          - "risk_override": dict of risk params to override
                          - "definition_patch": dict of top-level definition
                            keys to merge in (e.g. add an indicator, change
                            entry conditions). Applied on top of the saved
                            strategy definition for this run only via a
                            temporary strategy copy.
                          At minimum, supply [{"label": "baseline"}] to get
                          the base strategy metrics.
        base_risk:        Risk params shared by all variants (each variant's
                          risk_override is merged on top of this).
        starting_capital: Scale PnL for all variants (e.g. 150 for USDJPY).

    Returns a dict with:
        - "variants": list of {label, run_id, status} (all launched immediately)
        - "results":  list of {label, run_id, trades, total_pnl, per_trade_pnl,
                      win_rate, max_drawdown, sharpe_ratio} sorted by total_pnl desc
        - "best":     label of the top-performing variant
        - "analysis": plain-English interpretation of the results
    """
    import asyncio
    from database import async_session_factory
    from strategy.service import strategy_service
    from strategy.models import StrategyRunCreate
    from celery_app import enqueue

    # Launch all variants
    run_ids: list[tuple[str, int]] = []
    async with async_session_factory() as db:
        for v in variants:
            label = v.get("label", f"variant_{len(run_ids)}")
            risk = {**(base_risk or {}), **(v.get("risk_override") or {})}
            body = StrategyRunCreate(
                mode="backtest",
                dataset_id=dataset_id,
                risk_override=risk or None,
                starting_capital=starting_capital if starting_capital != 1.0 else None,
            )
            run = await strategy_service.create_run(db, strategy_id, body)
            await enqueue("execute_strategy_run", run.id)
            run_ids.append((label, run.id))

    launched = [{"label": lbl, "run_id": rid, "status": "pending"} for lbl, rid in run_ids]

    # Poll until all complete (max 10 min)
    deadline = 600
    poll_interval = 8
    elapsed = 0
    pending = dict(run_ids)

    while pending and elapsed < deadline:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        async with async_session_factory() as db:
            for label, rid in list(pending.items()):
                run = await strategy_service.get_run(db, strategy_id, rid)
                if run.status in ("completed", "error", "stopped"):
                    pending.pop(label)

    # Collect results
    results = []
    async with async_session_factory() as db:
        for label, rid in run_ids:
            try:
                m = await strategy_service.get_metrics(db, strategy_id, rid)
                trades = int(m.get("total_trades", 0))
                pnl = float(m.get("total_pnl", 0.0))
                results.append({
                    "label": label,
                    "run_id": rid,
                    "trades": trades,
                    "total_pnl": round(pnl, 4),
                    "per_trade_pnl": round(pnl / trades, 6) if trades else 0.0,
                    "win_rate": round(float(m.get("win_rate", 0.0)), 4),
                    "max_drawdown": round(float(m.get("max_drawdown", 0.0)), 4),
                    "sharpe_ratio": round(float(m.get("sharpe_ratio", 0.0)), 4),
                })
            except Exception:
                results.append({"label": label, "run_id": rid, "error": "failed"})

    results.sort(key=lambda r: r.get("total_pnl", float("-inf")), reverse=True)
    best = results[0]["label"] if results else "none"

    # Plain-English analysis
    lines = [f"Sweep of {len(variants)} variant(s) on strategy {strategy_id}, dataset {dataset_id}."]
    for r in results:
        if "error" in r:
            lines.append(f"  {r['label']}: ERROR")
        else:
            lines.append(
                f"  {r['label']}: {r['trades']} trades, PnL {r['total_pnl']:.4f}, "
                f"per-trade {r['per_trade_pnl']:.6f}, win {r['win_rate']:.1%}, "
                f"sharpe {r['sharpe_ratio']:.2f}"
            )
    lines.append(f"Best: {best}")

    return {"variants": launched, "results": results, "best": best, "analysis": "\n".join(lines)}


@mcp.tool()
async def register_webhook(url: str, events: list[str], secret: str) -> dict:
    """
    Register a webhook to receive platform events.
    Events: strategy.signal, trade.opened, trade.closed, strategy.run.completed,
            training.completed, dataset.ready, strategy.error.

    Args:
        url:    HTTP endpoint to POST events to.
        events: List of event names to subscribe to.
        secret: HMAC secret for payload verification.
    """
    from database import async_session_factory
    from webhooks.models import WebhookRegistration

    async with async_session_factory() as db:
        obj = WebhookRegistration(url=url, events=events, secret=secret)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return {"id": obj.id, "url": obj.url, "events": obj.events, "active": obj.active}
