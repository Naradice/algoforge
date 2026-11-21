"""
Patch live_runner.py with ML Filter support.
"""
from pathlib import Path

path = Path(r"c:\Users\ksato\workspace\finance\algoforge\backend\strategy\live_runner.py")
content = path.read_text(encoding="utf-8")

# 1. Parameter extraction
target1 = 'tol_yen = tolerance_pips * 0.01'
replacement1 = """tol_yen = tolerance_pips * 0.01

    ml_filter = params.get("ml_filter")
    ml_enabled = isinstance(ml_filter, dict) and ml_filter.get("enabled", False)
    ml_threshold = float(ml_filter.get("threshold", 0.50)) if ml_enabled else 0.0
    ml_model_path = ml_filter.get("model_path") if ml_enabled else None"""

assert target1 in content, "target1 not found"
content = content.replace(target1, replacement1, 1)

# 2. Check before order
target2 = 'if dir_to_enter:'
replacement2 = """if dir_to_enter and ml_enabled:
                            try:
                                from strategy.engine.rebound_features import compute_m30_indicators, predict_rebound_prob
                                candles_df = mt5_client.get_latest_candles(symbol, count=50)
                                if not candles_df.empty:
                                    df_ind = compute_m30_indicators(candles_df)
                                    last_bar = df_ind.iloc[-1]
                                    touch_p = bid if dir_to_enter == "buy" else ask
                                    prob = predict_rebound_prob(
                                        bar_series=last_bar,
                                        direction=dir_to_enter,
                                        exp_low=exp_low,
                                        exp_high=exp_high,
                                        touch_price=touch_p,
                                        bar_time=now_utc,
                                        model_path=ml_model_path,
                                    )
                                    if prob < ml_threshold:
                                        logger.info(f"[{strat_tag}] ML Filter rejected {dir_to_enter} entry: Rebound Prob {prob:.2f} < threshold {ml_threshold:.2f}")
                                        dir_to_enter = None
                                    else:
                                        entry_msg += f" [ML: {prob:.2f}]"
                                        logger.info(f"[{strat_tag}] ML Filter accepted {dir_to_enter} entry: Rebound Prob {prob:.2f} >= threshold {ml_threshold:.2f}")
                            except Exception as e:
                                logger.error(f"[{strat_tag}] ML Filter evaluation failed: {e}")

                        if dir_to_enter:"""

assert target2 in content, "target2 not found"
content = content.replace(target2, replacement2, 1)

path.write_text(content, encoding="utf-8")
print("Successfully patched live_runner.py with ML Filter support!")
