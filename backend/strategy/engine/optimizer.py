"""
Strategy Parameter Optimizer Engine.

Executes grid search over parameter ranges for:
- Standard risk parameters (sl_pct, tp_pct, etc.)
- Analyst forecast parameters (sl_pips, tp_pips, tolerance_pips, source)
Returns:
- top_results: Top N parameter combinations ranked by target_metric
- sensitivity_curve: Primary parameter vs metrics (PnL, win_rate, PF)
- all_results: Complete grid results
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np

from strategy.engine.backtest import run_backtest, _load_df


def run_parameter_optimization(
    strategy_definition: dict,
    artifact_path: str,
    param_ranges: dict[str, list[Any]],
    target_metric: str = "total_pnl",
    primary_param: str | None = None,
    starting_capital: float = 1_000_000.0,
) -> dict:
    """
    Runs a grid search across param_ranges.
    """
    if not param_ranges:
        return {"top_results": [], "sensitivity_curve": [], "all_results": []}

    # Normalize primary_param if not specified
    if primary_param is None or primary_param not in param_ranges:
        # Default priority: sl_pips -> sl_pct -> first param
        if "sl_pips" in param_ranges:
            primary_param = "sl_pips"
        elif "sl_pct" in param_ranges:
            primary_param = "sl_pct"
        else:
            primary_param = list(param_ranges.keys())[0]

    # Generate Cartesian product of all parameter ranges
    param_names = list(param_ranges.keys())
    value_lists = [param_ranges[k] for k in param_names]
    combinations = list(itertools.product(*value_lists))

    all_results = []

    is_analyst = strategy_definition.get("strategy_type", "").startswith("analyst_")

    for combo in combinations:
        param_dict = dict(zip(param_names, combo))
        current_def = dict(strategy_definition)
        current_params = dict(current_def.get("parameters", {}))

        if is_analyst:
            # Update analyst strategy parameters
            current_params.update(param_dict)
            current_def["parameters"] = current_params
        else:
            # Standard rule-based strategy: update risk params or indicator params
            risk = dict(current_def.get("risk", {}))
            for k, v in param_dict.items():
                if k in ("sl_pct", "tp_pct", "risk_pct", "slippage_pct", "commission_pct"):
                    risk[k] = v
                else:
                    current_params[k] = v
            current_def["risk"] = risk
            current_def["parameters"] = current_params

        try:
            trades, metrics, _ = run_backtest(
                current_def,
                artifact_path,
                starting_capital=starting_capital,
            )
            res_entry = {
                "params": param_dict,
                "trades": metrics.get("total_trades", len(trades)),
                "win_rate": round(metrics.get("win_rate", 0.0), 4),
                "total_pnl": round(metrics.get("total_pnl", 0.0), 2),
                "profit_factor": round(metrics.get("profit_factor", 0.0), 2),
                "max_drawdown": round(metrics.get("max_drawdown", 0.0), 4),
                "sharpe_ratio": round(metrics.get("sharpe_ratio", 0.0), 2),
            }
            all_results.append(res_entry)
        except Exception as e:
            # Skip failed variant
            continue

    if not all_results:
        return {"top_results": [], "sensitivity_curve": [], "all_results": []}

    # Sort results
    reverse_sort = target_metric in ("total_pnl", "win_rate", "profit_factor", "sharpe_ratio")
    sorted_results = sorted(all_results, key=lambda x: x.get(target_metric, 0.0), reverse=reverse_sort)
    top_results = sorted_results[:3]

    # Build sensitivity curve for primary_param
    # Group by primary_param and average or pick best of other params
    curve_dict: dict[Any, list[dict]] = {}
    for r in all_results:
        p_val = r["params"].get(primary_param)
        curve_dict.setdefault(p_val, []).append(r)

    sensitivity_curve = []
    # Sort curve points logically (None at end, numbers in ascending order)
    def curve_sort_key(k):
        if k is None or k == "None":
            return 999999.0
        try:
            return float(k)
        except (ValueError, TypeError):
            return 0.0

    sorted_keys = sorted(curve_dict.keys(), key=curve_sort_key)

    for k in sorted_keys:
        entries = curve_dict[k]
        # Pick the best PnL entry for this parameter value
        best_entry = max(entries, key=lambda x: x["total_pnl"])
        sensitivity_curve.append({
            "param_value": k,
            "param_label": f"{k} pips" if "pips" in primary_param and k is not None else ("No SL" if k is None else str(k)),
            "total_pnl": best_entry["total_pnl"],
            "win_rate": best_entry["win_rate"],
            "profit_factor": best_entry["profit_factor"],
            "trades": best_entry["trades"],
            "max_drawdown": best_entry["max_drawdown"],
        })

    return {
        "primary_param": primary_param,
        "target_metric": target_metric,
        "top_results": top_results,
        "sensitivity_curve": sensitivity_curve,
        "all_results": sorted_results,
    }
