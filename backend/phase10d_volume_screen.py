"""
Phase 10D: does traded volume (a liquidity / order-flow proxy) add information, or form a target
with nonlinear headroom?

Phases 10A-C used prices only. Dukascopy M1 candles carry the ECN's traded volume per minute
(dataset 158, USDJPY, bid candles, GMT). Aux inputs, each as its mean over the last 1/5/20/60 bars
up to the anchor:
  log_volume  log(volume)
  clv         close location value (2c - h - l) / (h - l), 0 when h == l -- where in the bar's range it
              closed: a crude signed-pressure proxy
  flow        clv * log_volume -- signed, volume-weighted pressure
Targets: volatility (future_log_rv h20/h60, vol_change h20, extreme h20), returns / direction
(h1/h5/h20) and volume itself (future_aux = mean log volume over the next h bars, h20/h60).
Every run scores linear_own (the same linear baseline without the aux inputs) on the same windows.

Usage: python phase10d_volume_screen.py [dataset_id]   (writes phase10d_volume_screen_<id>.json)
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

CONFIGS = [("future_log_rv", 20), ("future_log_rv", 60), ("vol_change", 20), ("extreme", 20),
           ("future_return", 1), ("future_return", 5), ("future_return", 20),
           ("direction", 1), ("direction", 5), ("direction", 20),
           ("future_aux", 20), ("future_aux", 60)]


def load(dataset_id: int, max_rows: int = 2_000_000) -> tuple[pd.Series, pd.DataFrame]:
    df = pd.read_parquet(_artifact_store() / asyncio.run(_dataset_artifact_path(dataset_id)))
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna(subset=["close"]).iloc[-max_rows:]
    df = df[df["volume"] > 0]
    rng = (df["high"] - df["low"]).to_numpy()
    clv = np.where(rng > 0, (2 * df["close"] - df["high"] - df["low"]).to_numpy() / np.where(rng > 0, rng, 1), 0.0)
    logv = np.log(df["volume"].to_numpy())
    aux = pd.DataFrame({"log_volume": logv, "clv": clv, "flow": clv * logv}, index=df.index)
    return df["close"], aux


def main():
    dataset_id = int(sys.argv[1]) if len(sys.argv) > 1 else 158
    out_path = Path(f"phase10d_volume_screen_{dataset_id}.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    close, aux = load(dataset_id)
    print(f"dataset {dataset_id}: {len(close)} rows {close.index[0]} .. {close.index[-1]}", flush=True)
    for target, h in CONFIGS:
        key = f"{target}_h{h}"
        if key in results:
            continue
        t0 = time.time()
        r = assess_target(close, target=target, horizon=h, obs=60, with_time=True, seed=0, aux=aux,
                          n_train=200_000)
        results[key] = r
        out_path.write_text(json.dumps(results, indent=1))
        m = r["metrics"]
        k = "auc" if r["task"] == "classification" else "r2"
        best = max((n for n in m if n not in ("base_rate", "persistence")), key=lambda n: m[n][k])
        print(f"{key:18s} {r['verdict']:18s} own {m['linear_own'][k]:+.4f} lin {m['linear'][k]:+.4f} "
              f"best {best} {m[best][k]:+.4f} hgb {m['hgb']['gain_rel']*100:+.1f}% mlp {m['mlp']['gain_rel']*100:+.1f}% "
              f"vol_gain {m['linear']['exog_gain_rel']*100:+.2f}% ci {[round(x, 6) for x in m['linear']['exog_ci95']]} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
