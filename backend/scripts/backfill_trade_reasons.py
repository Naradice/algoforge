"""Backfill entry_reason and detailed exit_reason for Strategy Runs."""

import asyncio
import datetime
from database import SessionLocal
from data.models import ReportForecast
from strategy.models import Trade, StrategyRun, Strategy

def backfill():
    with SessionLocal() as db:
        # Get all runs for Strategy 3 and 4
        runs = db.query(StrategyRun).filter(StrategyRun.strategy_id.in_([3, 4])).all()
        print(f"Found {len(runs)} runs for strategies 3 and 4.")

        # Load all forecasts for quick lookup
        forecasts = db.query(ReportForecast).filter(ReportForecast.symbol == "USDJPY").all()
        # map by date -> list of forecasts
        fc_by_date = {}
        for f in forecasts:
            fc_by_date.setdefault(f.report_date, []).append(f)

        for run in runs:
            trades = db.query(Trade).filter(Trade.run_id == run.id).all()
            if not trades:
                continue

            strat = db.query(Strategy).filter(Strategy.id == run.strategy_id).first()
            strat_type = (strat.definition or {}).get("strategy_type", "") if strat else ""

            updated_count = 0
            for t in trades:
                t_date = t.opened_at.date() if t.opened_at else None
                if not t_date:
                    continue

                # Find candidate forecast
                f_candidates = fc_by_date.get(t_date, [])
                if not f_candidates:
                    # check previous weekend / Friday
                    for diff in range(1, 5):
                        past_d = t_date - datetime.timedelta(days=diff)
                        if past_d in fc_by_date:
                            f_candidates = fc_by_date[past_d]
                            break

                # Match by entry_price
                matched_fc = None
                for fc in f_candidates:
                    if fc.expected_low is not None and abs(fc.expected_low - t.entry_price) < 0.05:
                        matched_fc = fc
                        break
                    if fc.expected_high is not None and abs(fc.expected_high - t.entry_price) < 0.05:
                        matched_fc = fc
                        break
                
                if not matched_fc and f_candidates:
                    matched_fc = f_candidates[0]

                source_name = matched_fc.source if matched_fc else "Report"
                exp_l = f"{matched_fc.expected_low:.2f}" if matched_fc and matched_fc.expected_low else f"{t.entry_price:.2f}"
                exp_h = f"{matched_fc.expected_high:.2f}" if matched_fc and matched_fc.expected_high else f"{t.entry_price:.2f}"

                # Determine entry_reason
                if "breakout" in strat_type.lower():
                    if t.direction == "buy":
                        t.entry_reason = f"Break High > {exp_h} ({source_name})"
                    else:
                        t.entry_reason = f"Break Low < {exp_l} ({source_name})"
                else:  # reversion
                    if t.direction == "buy":
                        t.entry_reason = f"Fade Low ≤ {exp_l} ({source_name})"
                    else:
                        t.entry_reason = f"Fade High ≥ {exp_h} ({source_name})"

                # Refine exit_reason if simple
                curr_exit = (t.exit_reason or "").lower()
                exit_p_str = f"{t.exit_price:.3f}" if t.exit_price is not None else ""
                if curr_exit in ("signal", "day_close", "close"):
                    t.exit_reason = f"signal (Close: {exit_p_str})"
                elif curr_exit in ("sl", "stop_loss"):
                    t.exit_reason = f"sl ({exit_p_str})"
                elif curr_exit in ("tp", "take_profit"):
                    t.exit_reason = f"tp ({exit_p_str})"

                updated_count += 1

            db.commit()
            print(f"Run {run.id} (Strategy {run.strategy_id}): Updated {updated_count} trades with entry_reason & exit_reason.")

if __name__ == "__main__":
    backfill()
