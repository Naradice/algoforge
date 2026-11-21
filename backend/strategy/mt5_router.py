"""
FastAPI Router for MetaTrader 5 (MT5) Configuration and Status.
"""

from __future__ import annotations

import logging
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from strategy.mt5_client import (
    MT5Config,
    load_mt5_config,
    save_mt5_config,
    mt5_client,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mt5", tags=["mt5"])


class MT5ConfigUpdate(BaseModel):
    login: int | None = None
    password: str | None = None
    server: str | None = None
    path: str | None = None
    magic: int | None = 100200
    default_lot: float | None = 0.1
    slippage: int | None = 20
    dry_run: bool | None = False


class MT5TestRequest(BaseModel):
    login: int | None = None
    password: str | None = None
    server: str | None = None
    path: str | None = None


@router.get("/config")
def get_config() -> dict[str, Any]:
    """Retrieve current MT5 configuration with password masked."""
    cfg = load_mt5_config()
    return {
        "login": cfg.login,
        "password_masked": "••••••••" if cfg.password else "",
        "server": cfg.server,
        "path": cfg.path,
        "magic": cfg.magic,
        "default_lot": cfg.default_lot,
        "slippage": cfg.slippage,
        "dry_run": cfg.dry_run,
    }


@router.post("/config")
def update_config(req: MT5ConfigUpdate) -> dict[str, Any]:
    """Update and persist MT5 configuration."""
    current = load_mt5_config()
    update_data = req.model_dump(exclude_unset=True)

    # Don't overwrite existing password with empty string
    if "password" in update_data and not update_data["password"]:
        del update_data["password"]

    for k, v in update_data.items():
        if hasattr(current, k):
            setattr(current, k, v)

    saved = save_mt5_config(current)
    mt5_client.config = saved
    # Re-connect with new config
    mt5_client.disconnect()
    ok, msg = mt5_client.connect()

    return {
        "message": "Configuration saved",
        "connected": ok,
        "status": msg,
        "config": {
            "login": saved.login,
            "server": saved.server,
            "path": saved.path,
            "default_lot": saved.default_lot,
            "dry_run": saved.dry_run,
        },
    }


@router.get("/account")
def get_account_status() -> dict[str, Any]:
    """Get live account status and current market ticks."""
    acc = mt5_client.get_account_info()
    tick = mt5_client.get_current_tick("USDJPY")
    return {
        "connected": acc is not None,
        "account": acc,
        "tick": tick,
        "dry_run": mt5_client.config.dry_run,
    }


@router.post("/test")
def test_connection(req: MT5TestRequest) -> dict[str, Any]:
    """Test connecting to MT5 with provided credentials."""
    from strategy.mt5_client import MT5Client, MT5Config

    temp_cfg = MT5Config(
        login=req.login if req.login else mt5_client.config.login,
        password=req.password if req.password else mt5_client.config.password,
        server=req.server if req.server else mt5_client.config.server,
        path=req.path if req.path else mt5_client.config.path,
    )
    client = MT5Client(temp_cfg)
    ok, msg = client.connect()
    acc = client.get_account_info() if ok else None

    # Keep global mt5_client synced
    if ok:
        mt5_client.config = temp_cfg
        mt5_client._connected = True
    else:
        # Fall back to trying to attach global client
        mt5_client.connect()

    if not ok:
        return {
            "success": False,
            "message": msg,
            "account": None,
        }

    return {
        "success": True,
        "message": msg,
        "account": acc,
    }
