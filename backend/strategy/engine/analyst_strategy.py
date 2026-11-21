"""
DEPRECATED: Analyst Forecast Strategy Compatibility Layer.

This module is deprecated. All analyst forecast and pivot reversion/breakout
strategies have been integrated into the standard AlgoForge backtest engine
(`strategy/engine/backtest.py`).

Execution now strictly enforces:
- Pending entry -> next bar Open execution (eliminating intra-bar lookahead)
- Fail-closed ML filtering via conditions.py
- Standard SL/TP evaluation and position lifecycle

Functions here are maintained solely for backwards compatibility.
"""
from __future__ import annotations

import logging
import warnings
from typing import Callable
import pandas as pd

from strategy.engine.forecast_levels import get_forecast_records, apply_forecast_indicators
from strategy.engine.backtest import run_backtest, _metrics_for as compute_metrics

logger = logging.getLogger("strategy.analyst_strategy")


def run_analyst_backtest(
    definition: dict,
    raw_df: pd.DataFrame,
    on_progress: Callable[[float], None] | None = None,
    starting_capital: float = 1_000_000.0,
) -> tuple[list[dict], dict, list[dict]]:
    """
    DEPRECATED: Forwarder to standard run_backtest.
    Saves DataFrame to a temporary path or passes definition to standard engine.
    """
    warnings.warn(
        "run_analyst_backtest is deprecated. Use strategy.engine.backtest.run_backtest instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.info("run_analyst_backtest called: delegating to standard backtest engine.")

    # In-memory artifact simulation: Save temporary parquet if needed, or pass path
    from pathlib import Path
    import tempfile

    temp_file = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        raw_df.to_parquet(temp_path)
        # Call standard run_backtest
        trades, metrics, curve = run_backtest(
            definition=definition,
            artifact_path=str(temp_path),
            on_progress=on_progress,
            starting_capital=starting_capital,
        )
        return trades, metrics, curve
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
