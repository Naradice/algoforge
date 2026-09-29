"""
Phase 10E: top-of-book microstructure -- does order flow carry information, or form a target with
nonlinear headroom?

Dataset 165: USDJPY Dukascopy ticks aggregated by the tick_aggregate datasource into 1-minute bars
(mid OHLC, spread, tick_count, top-of-book sizes, order-book imbalance, order-flow imbalance OFI;
2017-01..2023-03, GMT). Aux inputs, each as its mean over the last 1/5/20/60 bars up to the anchor
(plus hour x 20-bar-mean terms for the linear baseline):
  log_ticks   log tick count (quote activity)
  log_spread  log mean spread
  imbalance   mean (bid size - ask size) / total        (order-book imbalance)
  ofi         sign(OFI) * log1p(|OFI|)                  (Cont-Kukanov-Stoikov order-flow imbalance)
Targets on the mid price: returns / direction (h1/h5/h20), volatility (future_log_rv h20/h60).
Targets on the book itself (future_aux: mean over the next h bars of one aux series): log_ticks,
log_spread, ofi at h20.

Usage: python phase10e_microstructure_screen.py [dataset_id]   (writes phase10e_microstructure_screen.json)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model_core.analysis import assess_target
from probe_representations import _artifact_store, _dataset_artifact_path

AUX = ["log_ticks", "log_spread", "imbalance", "ofi"]
CONFIGS = [("future_return", 1, None), ("future_return", 5, None), ("future_return", 20, None),
           ("direction", 1, None), ("direction", 5, None), ("direction", 20, None),
           ("future_log_rv", 20, None), ("future_log_rv", 60, None),
           ("future_aux", 20, "log_ticks"), ("future_aux", 20, "log_spread"), ("future_aux", 20, "ofi")]


def load(dataset_id: int, max_rows: int = 2_000_000) -> tuple[pd.Series, pd.DataFrame]:
    df = pd.read_parquet(_artifact_store() / asyncio.run(_dataset_artifact_path(dataset_id)))
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna().iloc[-max_rows:]
    df = df[(df["spread"] > 0) & (df["tick_count"] > 0)]
    aux = pd.DataFrame({
        "log_ticks": np.log(df["tick_count"]), "log_spread": np.log(df["spread"]),
        "imbalance": df["imbalance"], "ofi": np.sign(df["ofi"]) * np.log1p(np.abs(df["ofi"])),
    }, index=df.index)
    return df["close"], aux


def main():
    dataset_id = int(sys.argv[1]) if len(sys.argv) > 1 else 165
    out_path = Path(f"phase10e_microstructure_screen_{dataset_id}.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    close, aux = load(dataset_id)
    print(f"dataset {dataset_id}: {len(close)} rows {close.index[0]} .. {close.index[-1]}", flush=True)
    for target, h, col in CONFIGS:
        key = f"{target}_h{h}" + (f"_{col}" if col else "")
        if key in results:
            continue
        t0 = time.time()
        a = aux[[col] + [c for c in AUX if c != col]] if col else aux     # future_aux reads column 0
        r = assess_target(close, target=target, horizon=h, obs=60, with_time=True, seed=0, aux=a,
                          n_train=200_000)
        results[key] = r
        out_path.write_text(json.dumps(results, indent=1))
        m = r["metrics"]
        k = "auc" if r["task"] == "classification" else "r2"
        best = max((n for n in m if n not in ("base_rate", "persistence")), key=lambda n: m[n][k])
        print(f"{key:26s} {r['verdict']:18s} own {m['linear_own'][k]:+.4f} lin {m['linear'][k]:+.4f} "
              f"best {best} {m[best][k]:+.4f} hgb {m['hgb']['gain_rel']*100:+.1f}% mlp {m['mlp']['gain_rel']*100:+.1f}% "
              f"aux_gain {m['linear']['exog_gain_rel']*100:+.2f}% ci {[round(x, 6) for x in m['linear']['exog_ci95']]} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
