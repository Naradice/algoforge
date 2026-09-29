"""
Phase 10A: beyond volatility -- which non-volatility USDJPY targets have signal, and nonlinear headroom?

Phases 8-9 showed USDJPY future volatility is persistence + intraday seasonality: a strong linear
model matches every nonlinear model, so pretraining has nothing to add there. Before any new
pretraining study, screen other downstream targets model-free with model_core.analysis.assess_target
(the same code the assess_target_difficulty MCP job runs):

  future_return  log return over the next h bars          (return prediction; reference 0)
  direction      up vs down over the next h bars          (flat futures dropped)
  extreme        any |r| in the next h above the series' 99.9% |r| quantile (absolute tail event)
  trend_er       efficiency ratio of the next h returns   (trend vs range regime)
  trend_change   future ER minus ER of the last h inputs  (regime transition)

Each is run on two independent USDJPY M1 sources over the same years: dataset 29 (2016-08..2022-08,
the Phase 8/9 data) and dataset 149 (HistData, most recent 2M rows ~ 2016..2022). Inputs: the last 60
log returns (+ time of day / weekday). Verdicts per docs/model-layer.md point 6.

Usage: python phase10a_target_screen.py <dataset_id> [max_rows]   (writes phase10a_target_screen_<id>.json)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd

from model_core.analysis import assess_target
from probe_representations import _artifact_store, _dataset_artifact_path

CONFIGS = [(t, h) for t in ("future_return", "direction") for h in (1, 5, 20, 60)] + \
          [(t, h) for t in ("extreme", "trend_er", "trend_change") for h in (20, 60)]


def load_close(dataset_id: int, max_rows: int) -> pd.Series:
    path = asyncio.run(_dataset_artifact_path(dataset_id))   # the API's DatasetRead omits artifact_path
    df = pd.read_parquet(_artifact_store() / path)
    df.columns = [c.lower() for c in df.columns]
    return df["close"].sort_index().dropna().iloc[-max_rows:]


def main():
    dataset_id = int(sys.argv[1])
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    out_path = Path(f"phase10a_target_screen_{dataset_id}.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    close = load_close(dataset_id, max_rows)
    print(f"dataset {dataset_id}: {len(close)} rows {close.index[0]} .. {close.index[-1]}", flush=True)
    for target, h in CONFIGS:
        key = f"{target}_h{h}"
        if key in results:
            continue
        t0 = time.time()
        r = assess_target(close, target=target, horizon=h, obs=60, with_time=True, seed=0)
        r["dataset_id"], r["rows"] = dataset_id, [str(close.index[0]), str(close.index[-1])]
        results[key] = r
        out_path.write_text(json.dumps(results, indent=1))
        lin = r["metrics"]["linear"]
        score = f"R2 {lin['r2']:+.4f}" if "r2" in lin else f"AUC {lin['auc']:.4f}"
        print(f"{key:20s} {r['verdict']:18s} {score} ci_vs_ref {lin['ci95_vs_reference']} "
              f"({time.time() - t0:.0f}s) -- {r['reason']}", flush=True)


if __name__ == "__main__":
    main()
