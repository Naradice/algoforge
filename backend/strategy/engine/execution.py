from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class OpenPosition:
    direction: str
    entry_price: float
    sl_price: float | None
    tp_price: float | None
    volume: float
    opened_at: datetime
    bar_index: int = 0
    mae: float = 0.0
    mfe: float = 0.0


@dataclass
class RiskParams:
    risk_type: str = "fixed"
    position_size: float = 1.0
    risk_pct: float = 0.01
    atr_multiplier: float = 2.0
    sl_pct: float = 0.02
    tp_pct: float = 0.04
    slippage_pct: float = 0.0005
    commission_pct: float = 0.001
    max_positions: int = 1
    daily_loss_limit_pct: float = 0.0
    cooldown_bars: int = 0
    trailing_stop: bool = False
    trailing_atr_multiplier: float = 3.0
    trailing_clip_with_price: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "RiskParams":
        return cls(
            risk_type=str(data.get("risk_type", "fixed")),
            position_size=float(data.get("position_size", 1.0)),
            risk_pct=float(data.get("risk_pct", 0.01)),
            atr_multiplier=float(data.get("atr_multiplier", 2.0)),
            sl_pct=float(data.get("sl_pct", 0.02)),
            tp_pct=float(data.get("tp_pct", 0.04)),
            slippage_pct=float(data.get("slippage_pct", 0.0005)),
            commission_pct=float(data.get("commission_pct", 0.001)),
            max_positions=int(data.get("max_positions", 1)),
            daily_loss_limit_pct=float(data.get("daily_loss_limit_pct", 0.0)),
            cooldown_bars=int(data.get("cooldown_bars", 0)),
            trailing_stop=bool(data.get("trailing_stop", False)),
            trailing_atr_multiplier=float(data.get("trailing_atr_multiplier", 3.0)),
            trailing_clip_with_price=bool(data.get("trailing_clip_with_price", False)),
        )


def calc_volume(rp: RiskParams, equity: float, entry_price: float, atr: float | None) -> float:
    if rp.risk_type == "percent_equity" and rp.sl_pct > 0:
        return (equity * rp.risk_pct) / (entry_price * rp.sl_pct)
    if rp.risk_type == "atr" and atr is not None and atr > 0:
        stop_dist = atr * rp.atr_multiplier
        return (equity * rp.risk_pct) / stop_dist
    return rp.position_size


def fill_price(mid: float, direction: str, slippage_pct: float) -> float:
    if direction == "buy":
        return mid * (1 + slippage_pct)
    return mid * (1 - slippage_pct)


def check_sl_tp(
    pos: OpenPosition,
    high: float,
    low: float,
    bar_dt: datetime,
    trades: list[dict],
    symbol: str,
    rp: RiskParams,
    phase: str | None = None,
) -> tuple[bool, OpenPosition | None]:
    if pos.direction == "buy":
        if pos.tp_price and high >= pos.tp_price:
            close_trade(pos, pos.tp_price, bar_dt, "tp", trades, symbol, rp, phase)
            return True, None
        if pos.sl_price and low <= pos.sl_price:
            close_trade(pos, pos.sl_price, bar_dt, "sl", trades, symbol, rp, phase)
            return True, None
    else:
        if pos.tp_price and low <= pos.tp_price:
            close_trade(pos, pos.tp_price, bar_dt, "tp", trades, symbol, rp, phase)
            return True, None
        if pos.sl_price and high >= pos.sl_price:
            close_trade(pos, pos.sl_price, bar_dt, "sl", trades, symbol, rp, phase)
            return True, None
    return False, pos


def close_trade(
    pos: OpenPosition,
    exit_price: float,
    closed_at: datetime,
    reason: str,
    trades: list[dict],
    symbol: str,
    rp: RiskParams,
    phase: str | None = None,
) -> float:
    if pos.direction == "buy":
        gross = (exit_price - pos.entry_price) / pos.entry_price * pos.volume
    else:
        gross = (pos.entry_price - exit_price) / pos.entry_price * pos.volume

    commission = rp.commission_pct * pos.volume
    profit = round(gross - commission, 6)

    mae_pct = pos.mae / pos.entry_price if pos.entry_price > 0 else 0.0
    mfe_pct = pos.mfe / pos.entry_price if pos.entry_price > 0 else 0.0

    trade = {
        "symbol": symbol,
        "direction": pos.direction,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "volume": pos.volume,
        "sl_price": pos.sl_price,
        "tp_price": pos.tp_price,
        "profit": profit,
        "opened_at": pos.opened_at,
        "closed_at": closed_at,
        "exit_reason": reason,
        "mae": round(mae_pct, 6),
        "mfe": round(mfe_pct, 6),
    }
    if phase is not None:
        trade["phase"] = phase
    trades.append(trade)
    return profit
