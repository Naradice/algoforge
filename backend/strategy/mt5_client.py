"""
MetaTrader 5 (MT5) Client & Execution Manager for AlgoForge.

Handles:
- Configuration management (login, password, server, path)
- Connection lifecycle (initialize, login, shutdown)
- Account information & balance monitoring
- Market data streaming (Tick, M1, M30)
- Order execution (Market Buy/Sell, SL, TP, Close Position)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5
import pandas as pd

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "mt5_config.json"


@dataclass
class MT5Config:
    login: int | None = None
    password: str | None = None
    server: str | None = None
    path: str | None = None
    magic: int = 100200
    default_lot: float = 0.1  # 0.1 lot = 10,000 units
    slippage: int = 20        # points / pips tolerance
    dry_run: bool = False      # if True, log orders without sending to broker


def load_mt5_config() -> MT5Config:
    """Load config from json or environment variables."""
    cfg = MT5Config()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    if hasattr(cfg, k):
                        if k == "login" and v:
                            setattr(cfg, k, int(v))
                        else:
                            setattr(cfg, k, v)
        except Exception as e:
            logger.warning(f"Failed to load {CONFIG_PATH}: {e}")

    # Env overrides
    if os.getenv("MT5_LOGIN"):
        cfg.login = int(os.getenv("MT5_LOGIN"))
    if os.getenv("MT5_PASSWORD"):
        cfg.password = os.getenv("MT5_PASSWORD")
    if os.getenv("MT5_SERVER"):
        cfg.server = os.getenv("MT5_SERVER")
    if os.getenv("MT5_PATH"):
        cfg.path = os.getenv("MT5_PATH")

    return cfg


def save_mt5_config(cfg: MT5Config | dict) -> MT5Config:
    """Save config to json file."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cfg, MT5Config):
        data = asdict(cfg)
    else:
        data = dict(cfg)

    # Convert login to int if possible
    if data.get("login"):
        try:
            data["login"] = int(data["login"])
        except ValueError:
            pass

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return load_mt5_config()


class MT5Client:
    """Wrapper around MetaTrader 5 API."""

    def __init__(self, config: MT5Config | None = None):
        self.config = config or load_mt5_config()
        self._connected = False

    def connect(self) -> tuple[bool, str]:
        """
        Connects to MT5 terminal using configured credentials.
        Returns (success, message).
        """
        init_kwargs: dict[str, Any] = {}
        if self.config.path:
            init_kwargs["path"] = self.config.path

        # If password is provided, attempt explicit login
        if self.config.login and self.config.server and self.config.password:
            init_kwargs["login"] = self.config.login
            init_kwargs["server"] = self.config.server
            init_kwargs["password"] = self.config.password

        success = mt5.initialize(**init_kwargs)
        if not success:
            # Fallback: try attaching to currently running MT5 terminal without credentials
            fallback_kwargs = {"path": self.config.path} if self.config.path else {}
            success = mt5.initialize(**fallback_kwargs)
            if not success:
                err = mt5.last_error()
                self._connected = False
                return False, f"mt5.initialize() failed: {err}"

        acc = mt5.account_info()
        if acc is None:
            err = mt5.last_error()
            self._connected = False
            return False, f"Failed to get account_info: {err}"

        self._connected = True
        return True, f"Connected to {acc.server} (Login: {acc.login}, Balance: {acc.balance:,.2f} {acc.currency})"

    def disconnect(self):
        """Shutdown MT5 connection."""
        try:
            mt5.shutdown()
        except Exception:
            pass
        self._connected = False

    def get_account_info(self) -> dict[str, Any] | None:
        """Fetch current account summary."""
        if not self._connected:
            ok, _ = self.connect()
            if not ok:
                return None

        acc = mt5.account_info()
        if acc is None:
            return None

        return {
            "login": acc.login,
            "server": acc.server,
            "company": acc.company,
            "currency": acc.currency,
            "balance": acc.balance,
            "equity": acc.equity,
            "profit": acc.profit,
            "margin": acc.margin,
            "margin_free": acc.margin_free,
            "margin_level": acc.margin_level,
            "leverage": acc.leverage,
            "trade_mode": acc.trade_mode,  # 0=demo, 1=contest, 2=real
        }

    def get_current_tick(self, symbol: str = "USDJPY") -> dict[str, Any] | None:
        """Get latest Bid/Ask tick."""
        if not self._connected:
            ok, _ = self.connect()
            if not ok:
                return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            "symbol": symbol,
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "spread": round((tick.ask - tick.bid) * 100, 1),  # in pips for JPY pairs
        }

    def get_latest_candles(
        self, symbol: str = "USDJPY", timeframe=mt5.TIMEFRAME_M30, count: int = 100
    ) -> pd.DataFrame:
        """Fetch latest N candles as DataFrame."""
        if not self._connected:
            self.connect()

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df

    def send_order(
        self,
        symbol: str,
        direction: str,  # "buy" or "sell"
        volume: float,   # lot size
        sl_price: float | None = None,
        tp_price: float | None = None,
        comment: str = "AlgoForge Live",
        magic: int | None = None,
    ) -> dict[str, Any]:
        """
        Executes a Market Order in MT5.
        Returns execution result dict.
        """
        if not self._connected:
            ok, msg = self.connect()
            if not ok:
                return {"success": False, "error": msg}

        if self.config.dry_run:
            logger.info(f"[DRY RUN] send_order: {direction.upper()} {volume} lots {symbol} SL={sl_price} TP={tp_price}")
            tick = mt5.symbol_info_tick(symbol)
            price = tick.ask if direction == "buy" else tick.bid
            return {
                "success": True,
                "order_id": 999999,
                "symbol": symbol,
                "direction": direction,
                "price": price,
                "volume": volume,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "comment": f"[DRY RUN] {comment}",
            }

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"success": False, "error": f"Symbol {symbol} not found in MT5"}

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return {"success": False, "error": f"Symbol {symbol} select failed"}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"success": False, "error": f"Cannot get tick for {symbol}"}

        order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == "buy" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(sl_price) if sl_price is not None else 0.0,
            "tp": float(tp_price) if tp_price is not None else 0.0,
            "deviation": int(self.config.slippage),
            "magic": int(magic) if magic is not None else int(self.config.magic),
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        res = mt5.order_send(request)
        if res is None:
            return {"success": False, "error": f"order_send failed: {mt5.last_error()}"}

        if res.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "success": False,
                "retcode": res.retcode,
                "comment": res.comment,
                "error": f"MT5 trade failed ({res.retcode}): {res.comment}",
            }

        return {
            "success": True,
            "order_id": res.order,
            "deal_id": res.deal,
            "symbol": symbol,
            "direction": direction,
            "price": res.price,
            "volume": res.volume,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "comment": comment,
        }

    def close_position(self, ticket: int, comment: str = "AlgoForge Close", magic: int | None = None) -> dict[str, Any]:
        """Close an open position by ticket number."""
        if not self._connected:
            self.connect()

        positions = mt5.positions_get(ticket=ticket)
        if not positions or len(positions) == 0:
            return {"success": False, "error": f"Position #{ticket} not found"}

        pos = positions[0]
        symbol = pos.symbol
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"success": False, "error": f"Cannot get tick for {symbol}"}

        direction = "sell" if pos.type == mt5.ORDER_TYPE_BUY else "buy"
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": pos.volume,
            "type": order_type,
            "price": float(price),
            "deviation": int(self.config.slippage),
            "magic": int(magic) if magic is not None else int(self.config.magic),
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        res = mt5.order_send(request)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            err = res.comment if res else str(mt5.last_error())
            return {"success": False, "error": f"Close position failed: {err}"}

        return {
            "success": True,
            "ticket": ticket,
            "close_price": res.price,
            "profit": pos.profit,
        }


# Singleton instance
mt5_client = MT5Client()
