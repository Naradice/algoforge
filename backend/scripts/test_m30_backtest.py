import pandas as pd
from strategy.engine.analyst_strategy import run_analyst_backtest

def main():
    m30_df = pd.read_parquet('artifacts/datasets/USDJPY_M30_latest.parquet')

    # 1. Breakout Momentum (Strategy 4)
    definition_bo = {
        'strategy_type': 'analyst_breakout_momentum',
        'symbol': 'USDJPY',
        'parameters': {'source': 'all', 'sl_pips': 15, 'tp_pips': 50}
    }
    trades_bo, metrics_bo, curve_bo = run_analyst_backtest(definition_bo, m30_df)
    print('=== Strategy 4: Breakout Momentum (M30 Strict Timing) ===')
    print(f"Total Trades: {metrics_bo['total_trades']}")
    print(f"Win Rate: {metrics_bo['win_rate']*100:.1f}%")
    print(f"Total PnL: {metrics_bo['total_pnl']:+,.0f} JPY")
    print(f"Profit Factor: {metrics_bo['profit_factor']:.2f}")
    print(f"Max Drawdown: {metrics_bo['max_drawdown']*100:.1f}%")
    
    print('\nSample Trades (First 5):')
    for t in trades_bo[:5]:
        print(f"  {t['opened_at']} -> {t['closed_at']} | {t['direction']} @ {t['entry_price']} -> {t['exit_price']} | {t['profit']:+,.0f} | {t['entry_reason']} | {t['exit_reason']}")

    # 2. Range Mean-Reversion (Strategy 3)
    definition_mr = {
        'strategy_type': 'analyst_range_reversion',
        'symbol': 'USDJPY',
        'parameters': {'source': 'SMBC', 'tolerance_pips': 10, 'sl_pips': 50, 'tp_pips': 80}
    }
    trades_mr, metrics_mr, curve_mr = run_analyst_backtest(definition_mr, m30_df)
    print('\n=== Strategy 3: Range Reversion (M30 Strict Timing) ===')
    print(f"Total Trades: {metrics_mr['total_trades']}")
    print(f"Win Rate: {metrics_mr['win_rate']*100:.1f}%")
    print(f"Total PnL: {metrics_mr['total_pnl']:+,.0f} JPY")
    print(f"Profit Factor: {metrics_mr['profit_factor']:.2f}")
    print(f"Max Drawdown: {metrics_mr['max_drawdown']*100:.1f}%")

    print('\nSample Trades (First 5):')
    for t in trades_mr[:5]:
        print(f"  {t['opened_at']} -> {t['closed_at']} | {t['direction']} @ {t['entry_price']} -> {t['exit_price']} | {t['profit']:+,.0f} | {t['entry_reason']} | {t['exit_reason']}")

if __name__ == "__main__":
    main()
