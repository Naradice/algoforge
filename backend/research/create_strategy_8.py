"""
Update Strategy 8 with ML-Filtered definition and trigger Backtest Run.
"""
from __future__ import annotations

import httpx

base_url = "http://127.0.0.1:8000/api/v1"

payload = {
    "name": "USD/JPY ML-Filtered FX168 Pivot Rebound",
    "description": "Daily mean-reversion within FX168 Daily Classic Pivot range filtered by LightGBM Bound-Rebound Classifier (prob >= 0.50). Optimized SL 15pips / TP 40pips / Tol 10pips.",
    "definition": {
        "strategy_type": "analyst_range_reversion",
        "symbol": "USDJPY",
        "parameters": {
            "source": "FX168",
            "sl_pips": 15,
            "tp_pips": 40,
            "tolerance_pips": 10,
            "lot_size": 0.1,
            "exit_at_close": True,
            "ml_filter": {
                "enabled": True,
                "threshold": 0.50,
                "model_path": "artifacts/models/rebound_classifier/lgbm_rebound_model.txt"
            }
        }
    }
}

res_put = httpx.put(f"{base_url}/strategies/8", json=payload)
print("Update Strategy 8:", res_put.status_code, res_put.json())

# Trigger Run
run_payload = {
    "mode": "backtest"
}
res_run = httpx.post(f"{base_url}/strategies/8/runs", json=run_payload)
print("Trigger Run:", res_run.status_code, res_run.json())
