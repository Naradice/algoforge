"""AI-driven strategy investigation using Gemini with function calling.

The agent calls two tools in a loop:
  get_strategy_info  — returns definition + recent run summaries
  run_sweep          — launches N risk-override variants and waits for all to complete

It streams SSE events: start | thinking | tool_call | tool_result | done | error
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

_SYSTEM = """You are a quantitative trading strategy analyst. Your job is to investigate
a strategy's profitability after realistic transaction costs and suggest concrete improvements.

Available tools:
  get_strategy_info  — call this FIRST to understand the strategy's indicators, conditions,
                       risk params, and recent run history.
  run_sweep          — runs multiple backtest variants in parallel (risk parameter changes
                       only; definition changes are recommended as text). Returns ranked
                       results with per-trade gross edge.

Workflow:
1. Call get_strategy_info.
2. Design 4–6 variants that test the most promising improvements:
   - baseline: realistic costs, no other changes
   - wider trailing stop (trailing_atr_multiplier: 5 or 6)
   - zero costs (to measure gross edge ceiling)
   - stricter costs (to find the cost break-even point)
   - adjusted position sizing if relevant
3. Call run_sweep with all variants at once.
4. Analyse results: compute gross edge per trade, compare against round-trip cost,
   identify the best variant and the cost break-even threshold.
5. Write a concise report:
   - Summary table of variants (trades, PnL, per-trade PnL)
   - Which parameter change helped most and why
   - Per-trade edge vs cost analysis (is the strategy viable?)
   - Concrete next steps (definition changes to try, e.g. EMA trend filter)

Realistic OANDA FX costs for USDJPY: slippage_pct 0.00004, commission_pct 0.
Be direct and quantitative. No vague advice."""


def _sse(event_type: str, **kwargs) -> str:
    return f"data: {json.dumps({'type': event_type, **kwargs}, default=str)}\n\n"


async def stream_investigation(strategy_id: int, dataset_id: int) -> AsyncGenerator[str, None]:
    """Yield SSE data lines for an AI-driven strategy investigation."""
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError:
        yield _sse("error", content="google-genai not installed — set GOOGLE_API_KEY")
        return

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        yield _sse("error", content="GOOGLE_API_KEY is not set")
        return

    client = genai.Client(api_key=api_key)

    tools = [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_strategy_info",
            description=(
                "Return the strategy's full definition (indicators, entry/exit conditions, "
                "risk params) and a summary of recent backtest runs."
            ),
            parameters=types.Schema(type=types.Type.OBJECT, properties={}, required=[]),
        ),
        types.FunctionDeclaration(
            name="run_sweep",
            description=(
                "Launch multiple backtest variants in parallel (risk param overrides only) "
                "and wait for all to complete. Returns a ranked comparison table."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "variants": types.Schema(
                        type=types.Type.ARRAY,
                        description=(
                            "List of variants. Each has 'label' (str) and optionally "
                            "'risk_override' (dict of risk params to override)."
                        ),
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "label": types.Schema(type=types.Type.STRING),
                                "risk_override": types.Schema(type=types.Type.OBJECT),
                            },
                            required=["label"],
                        ),
                    ),
                },
                required=["variants"],
            ),
        ),
    ])]

    messages = [types.Content(role="user", parts=[types.Part(text=(
        f"Investigate strategy {strategy_id} on dataset {dataset_id}. "
        "Find the best risk parameter configuration and assess whether the strategy "
        "is profitable after realistic transaction costs. Give concrete recommendations."
    ))])]

    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM,
        tools=tools,
        temperature=0.2,
    )

    yield _sse("start", content="AI investigation started…")

    for _round in range(8):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL, contents=messages, config=config
            )
        except Exception as exc:
            yield _sse("error", content=f"LLM call failed: {exc}")
            return

        candidate = response.candidates[0]
        messages.append(candidate.content)

        # Stream any text the model produced
        for part in candidate.content.parts:
            if getattr(part, "text", None):
                yield _sse("thinking", content=part.text)

        # Collect function calls
        calls = [p.function_call for p in candidate.content.parts
                 if getattr(p, "function_call", None)]
        if not calls:
            break  # No more tool calls — agent is done

        tool_parts = []
        for fc in calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}
            yield _sse("tool_call", tool=name)

            try:
                result = await _execute_tool(name, args, strategy_id, dataset_id)
                yield _sse("tool_result", tool=name, content=_summarize(name, result))
            except Exception as exc:
                result = {"error": str(exc)}
                yield _sse("tool_result", tool=name, content=f"Error: {exc}")

            tool_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=name,
                    response={"result": result},
                )
            ))

        messages.append(types.Content(role="user", parts=tool_parts))

    yield _sse("done", content="Investigation complete")


# ── Tool implementations ───────────────────────────────────────────────────────

async def _execute_tool(name: str, args: dict, strategy_id: int, dataset_id: int) -> dict:
    from database import async_session_factory
    from strategy.service import strategy_service
    from strategy.models import StrategyRunCreate
    from celery_app import enqueue

    if name == "get_strategy_info":
        async with async_session_factory() as db:
            strategy = await strategy_service.get_strategy(db, strategy_id)
            runs, _ = await strategy_service.list_runs(db, strategy_id, limit=5)
            summaries = []
            for r in runs:
                try:
                    m = await strategy_service.get_metrics(db, strategy_id, r.id)
                except Exception:
                    m = {}
                summaries.append({
                    "run_id": r.id,
                    "dataset_id": r.dataset_id,
                    "trades": m.get("total_trades"),
                    "total_pnl": m.get("total_pnl"),
                    "win_rate": m.get("win_rate"),
                    "risk_override": r.risk_override,
                })
        return {"definition": strategy.definition, "recent_runs": summaries}

    if name == "run_sweep":
        variants = args.get("variants", [])
        run_map: list[tuple[str, int]] = []

        async with async_session_factory() as db:
            for v in variants:
                label = v.get("label", f"v{len(run_map)}")
                risk = v.get("risk_override") or None
                run = await strategy_service.create_run(db, strategy_id, StrategyRunCreate(
                    mode="backtest", dataset_id=dataset_id, risk_override=risk,
                ))
                await enqueue("execute_strategy_run", run.id)
                run_map.append((label, run.id))

        # Poll until all complete (max 10 min)
        pending = {rid for _, rid in run_map}
        for _ in range(75):          # 75 × 8 s = 600 s
            await asyncio.sleep(8)
            if not pending:
                break
            async with async_session_factory() as db:
                for rid in list(pending):
                    r = await strategy_service.get_run(db, strategy_id, rid)
                    if r.status in ("completed", "error", "stopped"):
                        pending.discard(rid)

        # Collect results
        results = []
        async with async_session_factory() as db:
            for label, rid in run_map:
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
                        "win_rate": round(float(m.get("win_rate", 0)), 4),
                        "sharpe": round(float(m.get("sharpe_ratio", 0)), 4),
                    })
                except Exception as exc:
                    results.append({"label": label, "run_id": rid, "error": str(exc)})

        results.sort(key=lambda r: r.get("total_pnl", float("-inf")), reverse=True)
        return {"results": results, "best": results[0]["label"] if results else "none"}

    return {"error": f"Unknown tool: {name}"}


def _summarize(tool: str, result: dict) -> str:
    if tool == "get_strategy_info":
        defn = result.get("definition", {})
        inds = [i.get("id") for i in defn.get("indicators", [])]
        runs = result.get("recent_runs", [])
        last = runs[0] if runs else {}
        return (
            f"Indicators: {inds}. "
            f"Most recent run: {last.get('trades')} trades, PnL {last.get('total_pnl')}."
        )
    if tool == "run_sweep":
        lines = [f"Best variant: {result.get('best')}"]
        for r in result.get("results", [])[:6]:
            if "error" in r:
                lines.append(f"  {r['label']}: ERROR")
            else:
                lines.append(
                    f"  {r['label']}: {r['trades']} trades  "
                    f"PnL {r['total_pnl']:.4f}  "
                    f"per-trade {r['per_trade_pnl']:.6f}  "
                    f"win {r.get('win_rate', 0):.1%}"
                )
        return "\n".join(lines)
    return str(result)[:400]
