"""
WebSocket router — real-time chat for strategy runs.

Endpoints:
    WS /api/v1/ws/strategies/{strategy_id}/runs/{run_id}/chat
        — Bidirectional chat with Gemini AI agent.
          Client sends:  {"message": "..."}
          Server sends:  {"role": "agent", "content": "...", "is_final": true}
                    or:  {"error": "..."}

Mount in main.py:
    from ws_router import ws_router
    app.include_router(ws_router, prefix="/api/v1")
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

ws_router = APIRouter(tags=["websocket"])


@ws_router.websocket("/ws/strategies/{strategy_id}/runs/{run_id}/chat")
async def chat_ws(websocket: WebSocket, strategy_id: int, run_id: int) -> None:
    from database import async_session_factory
    from strategy.models import Strategy, StrategyRun
    from strategy.repository import strategy_repo
    from strategy.chat_agent import stream_response
    from sqlalchemy import select

    await websocket.accept()

    # Load strategy + run for context
    async with async_session_factory() as db:
        r = await db.execute(select(StrategyRun).where(StrategyRun.id == run_id))
        run = r.scalar_one_or_none()
        if run is None or run.strategy_id != strategy_id:
            await websocket.close(code=4004)
            return
        r2 = await db.execute(select(Strategy).where(Strategy.id == strategy_id))
        strategy = r2.scalar_one_or_none()
        if strategy is None:
            await websocket.close(code=4004)
            return

    try:
        while True:
            data = await websocket.receive_json()
            user_message = str(data.get("message", "")).strip()
            if not user_message:
                continue

            # Persist user message
            async with async_session_factory() as db:
                await strategy_repo.add_chat_message(
                    db, run_id=run_id, role="user", message=user_message
                )

            # Load context for AI
            async with async_session_factory() as db:
                metrics = await strategy_repo.get_metrics(db, run_id)
                trades = await strategy_repo.get_trades(db, run_id)
                history = await strategy_repo.get_chat_history(db, run_id)

            # Stream AI response
            full_response = ""
            async for chunk in stream_response(
                strategy_definition=strategy.definition,
                metrics=metrics,
                recent_trades=trades[-10:],
                chat_history=history[-10:],
                user_message=user_message,
            ):
                full_response += chunk

            await websocket.send_json({
                "role": "agent",
                "content": full_response,
                "is_final": True,
            })

            # Persist AI response
            if full_response:
                async with async_session_factory() as db:
                    await strategy_repo.add_chat_message(
                        db, run_id=run_id, role="agent", message=full_response
                    )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
