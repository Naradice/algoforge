"""

Live & Paper Trading Runner for AlgoForge with MetaTrader 5 (MT5).



Supports:

1. Analyst Forecast Strategies (Range Reversion & Breakout Momentum)

2. Standard Rule/Condition Strategies

3. Real broker execution via MT5 or simulated paper execution

"""



from __future__ import annotations



import asyncio

import logging

import os

from datetime import datetime, timedelta, timezone

from typing import Any



from database import async_session_factory

from strategy.models import StrategyRun, Trade, RunMetric

from strategy.mt5_client import mt5_client
from strategy.engine.backtest import _normalise
from strategy.engine.conditions import evaluate_conditions
from strategy.engine.forecast_levels import ANALYST_CLOSE_UTC
from strategy.engine.indicators import apply_indicators

from sqlalchemy import select, update



logger = logging.getLogger(__name__)



# Polling interval in seconds for live market checking

POLL_INTERVAL_S = int(os.getenv("LIVE_POLL_INTERVAL_S", "5"))





async def run_live(

    run_id: int,

    strategy_definition: dict,

    model_cache: dict | None = None,

    mode: str = "live",  # "live" or "paper"

    risk_override: dict | None = None,

    strategy_id: int | None = None,

) -> dict[str, Any]:

    """

    Main live/paper execution loop.

    Connects to MT5, monitors market rates, evaluates signals, and places trades.

    """

    strategy_type = strategy_definition.get("strategy_type", "")

    symbol = strategy_definition.get("symbol", "USDJPY")

    params = dict(strategy_definition.get("parameters", {}))



    # Apply any risk_override parameters

    if risk_override:

        for k, v in risk_override.items():

            if v is not None:

                params[k] = v



    # Lot size: from strategy params, or from MT5 config

    lot_size = float(params.get("lot_size", mt5_client.config.default_lot) or 0.1)

    definition = _normalise({**strategy_definition, "parameters": params})
    risk = definition.get("risk", {})
    pip_size = risk.get("pip_size", 0.01)
    sl_yen = float(risk.get("sl_pips") or 0) * pip_size
    tp_yen = float(risk.get("tp_pips") or 0) * pip_size
    opposite_tp = str(params.get("tp_pips")).lower() == "opposite"
    time_stop = params.get("exit_at_close", True)
    entered_on = {}
    last_signal_bar = None

    logger.info(f"Starting {mode.upper()} run #{run_id} for {symbol} ({strategy_type})")



    # Order comment tag & MT5 magic number identifying strategy & run

    strat_tag = f"S{strategy_id or '?'}_R{run_id}"

    magic_num = 100000 + ((strategy_id or 0) % 1000) * 1000 + (run_id % 1000)



    # 1. Connect to MT5

    ok, msg = mt5_client.connect()

    if not ok:

        async with async_session_factory() as db:

            await db.execute(

                update(StrategyRun).where(StrategyRun.id == run_id).values(

                    status="error",

                    message=f"MT5 Connection Failed: {msg}",

                    ended_at=datetime.now(timezone.utc),

                )

            )

            await db.commit()

        return {"error": msg}



    acc = mt5_client.get_account_info()

    acc_desc = f"{acc['server']} (Login: {acc['login']}, Balance: {acc['balance']:,.0f} {acc['currency']})" if acc else "MT5"



    async with async_session_factory() as db:

        await db.execute(

            update(StrategyRun).where(StrategyRun.id == run_id).values(

                status="running",

                message=f"Active on {acc_desc} | Mode: {mode.upper()} | Lot: {lot_size}",

                started_at=datetime.now(timezone.utc),

            )

        )

        await db.commit()



    active_trade: dict[str, Any] | None = None

    trades_count = 0

    total_pnl = 0.0



    try:

        while True:

            # Check if run was stopped via UI/API

            async with async_session_factory() as db:

                res = await db.execute(select(StrategyRun).where(StrategyRun.id == run_id))

                run_rec = res.scalar_one_or_none()



            if run_rec is None or run_rec.status in ("stopped", "completed", "error"):

                logger.info(f"Run #{run_id} status changed to {run_rec.status if run_rec else 'None'}, stopping loop.")

                break



            # Fetch current rate from MT5

            tick = mt5_client.get_current_tick(symbol)

            if tick is None:

                await asyncio.sleep(POLL_INTERVAL_S)

                continue



            bid = float(tick["bid"])

            ask = float(tick["ask"])

            now_utc = datetime.now(timezone.utc)



            # =================================================================

            # A. Position Management (Check if active position should close)

            # =================================================================

            if active_trade is not None:

                ticket = active_trade.get("ticket")

                direction = active_trade.get("direction")

                entry_p = active_trade.get("entry_price")

                trade_db_id = active_trade.get("db_id")



                should_close = False

                close_reason = "signal"

                close_price = bid if direction == "buy" else ask



                # Check SL / TP

                if direction == "buy":

                    if sl_yen and bid <= (entry_p - sl_yen):

                        should_close = True

                        close_reason = f"sl ({bid:.3f})"

                    elif tp_yen and bid >= (entry_p + tp_yen):

                        should_close = True

                        close_reason = f"tp ({bid:.3f})"

                else:  # sell

                    if sl_yen and ask >= (entry_p + sl_yen):

                        should_close = True

                        close_reason = f"sl ({ask:.3f})"

                    elif tp_yen and ask <= (entry_p - tp_yen):

                        should_close = True

                        close_reason = f"tp ({ask:.3f})"



                # Check NY Close / Day End (23:30 UTC)

                if time_stop and (now_utc.time() >= ANALYST_CLOSE_UTC or
                                  active_trade["opened_at"].date() != now_utc.date()):

                    should_close = True

                    close_reason = f"Day Close ({close_price:.3f})"



                if should_close:

                    logger.info(f"Closing position #{ticket} due to {close_reason}")

                    if mode == "live" and ticket:

                        close_comment = f"{strat_tag} {close_reason}"[:31]

                        c_res = mt5_client.close_position(ticket, comment=close_comment, magic=magic_num)

                        if not c_res.get("success"):
                            logger.warning("[%s] Close failed for position %s", strat_tag, ticket)
                            await asyncio.sleep(POLL_INTERVAL_S)
                            continue
                        close_price = c_res.get("close_price", close_price)



                    pnl = (close_price - entry_p) * 100_000.0 * lot_size if direction == "buy" else (entry_p - close_price) * 100_000.0 * lot_size

                    total_pnl += pnl



                    # Update Trade in DB

                    async with async_session_factory() as db:

                        await db.execute(

                            update(Trade).where(Trade.id == trade_db_id).values(

                                exit_price=close_price,

                                closed_at=now_utc,

                                profit=pnl,

                                exit_reason=f"[{strat_tag}] {close_reason}",

                            )

                        )

                        await db.commit()



                    active_trade = None



            # =================================================================

            # B. Signal Evaluation (Analyst Forecast Strategies)

            # =================================================================

            dir_to_enter = None
            if strategy_type.startswith("analyst_") and now_utc.time() < ANALYST_CLOSE_UTC:
                bar_open = now_utc.replace(minute=(now_utc.minute // 30) * 30, second=0, microsecond=0)
                signal_bar = bar_open - timedelta(minutes=30)
                # Do not enter from an old signal after starting mid-bar.
                entry_deadline = bar_open + timedelta(seconds=2 * POLL_INTERVAL_S)
                if (signal_bar != last_signal_bar and signal_bar.date() == bar_open.date()
                        and now_utc < entry_deadline):
                    if active_trade is not None:
                        last_signal_bar = signal_bar
                    else:
                        try:
                            candles = mt5_client.get_latest_candles(symbol, count=50)
                            if candles is not None and signal_bar in candles.index:
                                last_signal_bar = signal_bar
                                closed = candles.sort_index().loc[:signal_bar]
                                row = apply_indicators(closed, definition["indicators"]).iloc[-1]
                                for side, direction in (("long", "buy"), ("short", "sell")):
                                    if entered_on.get(direction) != bar_open.date() and evaluate_conditions(
                                        row, definition[side]["entry"]
                                    ):
                                        dir_to_enter = direction
                                        break
                        except Exception:
                            logger.exception("[%s] Closed M30 signal evaluation failed", strat_tag)

            if dir_to_enter:
                is_buy = dir_to_enter == "buy"
                entry_quote = ask if is_buy else bid
                sign = 1 if is_buy else -1
                target_sl = entry_quote - sign * sl_yen if sl_yen else None
                target_tp = entry_quote + sign * tp_yen if tp_yen else None
                if opposite_tp:
                    boundary = float(row["expected_high" if is_buy else "expected_low"])
                    target_tp = boundary if sign * (boundary - entry_quote) > 0 else target_tp
                entry_msg = f"Closed M30 {dir_to_enter} signal"

                order_comment = f"{strat_tag} {entry_msg}"[:31]

                order_res = mt5_client.send_order(

                    symbol=symbol,

                    direction=dir_to_enter,

                    volume=lot_size,

                    sl_price=target_sl,

                    tp_price=target_tp,

                    comment=order_comment,

                    magic=magic_num,

                )

                if order_res.get("success"):

                    trades_count += 1
                    entered_on[dir_to_enter] = now_utc.date()

                    fill_price = order_res.get("price", ask if dir_to_enter == "buy" else bid)

                    order_id = order_res.get("order_id", 0)



                    async with async_session_factory() as db:

                        trade_obj = Trade(

                            run_id=run_id,

                            symbol=symbol,

                            direction=dir_to_enter,

                            entry_price=fill_price,

                            volume=lot_size,

                            sl_price=target_sl,

                            tp_price=target_tp,

                            opened_at=now_utc,

                            entry_reason=f"[{strat_tag}] {entry_msg}",

                            phase="live",

                        )

                        db.add(trade_obj)

                        await db.flush()

                        db_id = trade_obj.id

                        await db.commit()



                    active_trade = {

                        "ticket": order_id,

                        "direction": dir_to_enter,

                        "entry_price": fill_price,

                        "db_id": db_id,

                        "opened_at": now_utc,

                    }

                    logger.info(f"Entered {dir_to_enter.upper()} @ {fill_price:.3f} | {entry_msg}")



            # Update progress message

            async with async_session_factory() as db:

                msg_status = f"{symbol} Bid={bid:.3f} Ask={ask:.3f} | Pos: {active_trade['direction'].upper() if active_trade else 'None'} | Trades: {trades_count} | PnL: {total_pnl:+,.0f} JPY"

                await db.execute(

                    update(StrategyRun).where(StrategyRun.id == run_id).values(message=msg_status)

                )

                await db.commit()



            await asyncio.sleep(POLL_INTERVAL_S)



    finally:

        # Close any open position on stop

        if active_trade is not None:

            ticket = active_trade.get("ticket")

            direction = active_trade.get("direction")

            entry_p = active_trade.get("entry_price")

            trade_db_id = active_trade.get("db_id")

            now_utc = datetime.now(timezone.utc)

            tick = mt5_client.get_current_tick(symbol)

            close_price = tick["bid"] if direction == "buy" and tick else (tick["ask"] if tick else entry_p)



            if mode == "live" and ticket:

                mt5_client.close_position(ticket, comment=f"{strat_tag} Stop", magic=magic_num)



            pnl = (close_price - entry_p) * 100_000.0 * lot_size if direction == "buy" else (entry_p - close_price) * 100_000.0 * lot_size

            total_pnl += pnl



            async with async_session_factory() as db:

                await db.execute(

                    update(Trade).where(Trade.id == trade_db_id).values(

                        exit_price=close_price,

                        closed_at=now_utc,

                        profit=pnl,

                        exit_reason="stopped",

                    )

                )

                await db.commit()



        # Update final metrics

        async with async_session_factory() as db:

            await db.execute(

                update(StrategyRun).where(StrategyRun.id == run_id).values(

                    status="stopped",

                    ended_at=datetime.now(timezone.utc),

                    message=f"Stopped | Total Trades: {trades_count} | PnL: {total_pnl:+,.0f} JPY",

                )

            )

            db.add(RunMetric(run_id=run_id, key="total_trades", value=float(trades_count)))

            db.add(RunMetric(run_id=run_id, key="total_pnl", value=float(total_pnl)))

            await db.commit()



    return {"trades": trades_count, "total_pnl": total_pnl}





# Backward-compatible run_paper

async def run_paper(run_id: int, strategy_definition: dict, model_cache: dict) -> dict[str, Any]:

    return await run_live(run_id, strategy_definition, model_cache=model_cache, mode="paper")
