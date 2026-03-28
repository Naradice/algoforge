"""Strategy configuration / schema endpoints."""
from __future__ import annotations

from fastapi import APIRouter
from schemas import DataResponse

config_router = APIRouter(prefix="/strategy-config", tags=["strategy-config"])


@config_router.get("/handlers")
async def list_handlers():
    handlers = [
        {
            "name": "comparison",
            "description": "Standard comparison condition: left op right (e.g. rsi < 70)",
            "params_schema": {"left": "str", "op": "str", "right": "str|float"},
        },
        {
            "name": "ml_signal",
            "description": "ML model signal condition: model predicts buy/sell direction",
            "params_schema": {"type": "ml_signal", "model_id": "int", "direction": "str", "step": "int", "min_confidence": "float"},
        },
        {
            "name": "llm_signal",
            "description": "LLM signal condition: language model analyzes recent bars",
            "params_schema": {"type": "llm_signal", "direction": "str", "lookback": "int", "model": "str", "columns": "list[str]"},
        },
    ]
    return DataResponse(data=handlers)


@config_router.get("/event-types")
async def list_event_types():
    event_types = [
        {"name": "tick", "description": "New price tick received"},
        {"name": "signal", "description": "Entry/exit condition evaluated"},
        {"name": "trade_opened", "description": "New position opened"},
        {"name": "trade_closed", "description": "Position closed"},
        {"name": "condition_result", "description": "Individual condition evaluation result"},
        {"name": "error", "description": "Strategy execution error"},
    ]
    return DataResponse(data=event_types)


@config_router.get("/risk-options")
async def list_risk_options():
    risk_options = [
        {
            "name": "fixed_size",
            "description": "Fixed position size (position_size param)",
            "config_schema": {"position_size": "float", "sl_pct": "float", "tp_pct": "float"},
        },
    ]
    return DataResponse(data=risk_options)


@config_router.get("/broker-clients")
async def list_broker_clients():
    clients = [
        {"name": "yfinance", "description": "Yahoo Finance (paper/backtest only)", "credentials_required": []},
        {"name": "mt5", "description": "MetaTrader 5", "credentials_required": ["login", "password", "server"]},
        {"name": "coincheck", "description": "Coincheck crypto exchange", "credentials_required": ["access_key", "secret_key"]},
    ]
    return DataResponse(data=clients)
