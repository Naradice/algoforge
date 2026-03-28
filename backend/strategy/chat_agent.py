"""
Gemini-powered chat agent for strategy run conversations.

Provides streaming AI responses with full strategy context (definition,
metrics, recent trades, chat history).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from typing import Any

_SYSTEM_PROMPT = """You are a quantitative trading analyst assistant embedded in AlgoForge.
You have full context of the strategy being discussed: its definition, live or backtest metrics,
and recent trade history. Help the user understand performance, identify weaknesses, suggest
parameter improvements, and explain individual signals. Be concise, specific, and data-driven.
Never fabricate numbers — only reference the data provided."""


async def stream_response(
    strategy_definition: dict[str, Any],
    metrics: dict[str, Any],
    recent_trades: list[Any],
    chat_history: list[Any],
    user_message: str,
) -> AsyncGenerator[str, None]:
    """
    Yield text chunks of the AI response.
    Falls back to a plain error message if google-genai is unavailable.
    """
    try:
        from google import genai  # type: ignore
    except ImportError:
        yield "google-genai is not installed. Install it to enable AI chat."
        return

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        yield "No GOOGLE_API_KEY is configured. AI chat is unavailable."
        return

    client = genai.Client(api_key=api_key)
    model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash")

    # Build context block
    trades_summary = [
        {
            "symbol": t.symbol if hasattr(t, "symbol") else t.get("symbol"),
            "direction": t.direction if hasattr(t, "direction") else t.get("direction"),
            "profit": t.profit if hasattr(t, "profit") else t.get("profit"),
            "opened_at": str(t.opened_at if hasattr(t, "opened_at") else t.get("opened_at")),
        }
        for t in recent_trades
    ]

    context = (
        f"Strategy definition:\n{json.dumps(strategy_definition, indent=2)}\n\n"
        f"Performance metrics:\n{json.dumps(metrics, indent=2)}\n\n"
        f"Recent trades (latest {len(trades_summary)}):\n{json.dumps(trades_summary, indent=2, default=str)}"
    )

    # Contents array for Gemini multi-turn
    contents: list[dict] = [
        {"role": "user", "parts": [{"text": _SYSTEM_PROMPT + "\n\n" + context}]},
        {"role": "model", "parts": [{"text": "Understood. I have the strategy context loaded and am ready to assist."}]},
    ]

    for msg in chat_history[-8:]:
        role = "user" if (msg.role if hasattr(msg, "role") else msg.get("role")) == "user" else "model"
        message = msg.message if hasattr(msg, "message") else msg.get("message", "")
        contents.append({"role": role, "parts": [{"text": message}]})

    contents.append({"role": "user", "parts": [{"text": user_message}]})

    try:
        # Use sync streaming in a thread to avoid async API compatibility issues
        full_text = await asyncio.to_thread(_call_gemini_sync, client, model_name, contents)
        yield full_text
    except Exception as e:
        yield f"[AI error: {e}]"


def _call_gemini_sync(client: Any, model_name: str, contents: list) -> str:
    response = client.models.generate_content(model=model_name, contents=contents)
    return response.text or ""
