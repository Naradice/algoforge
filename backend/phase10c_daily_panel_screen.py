"""
Phase 10C: long horizons across asset classes -- daily bars, days-to-weeks targets.

Phases 8-10B found nothing but volatility (persistence + seasonality) in USDJPY M1, with or without
the other currency pairs. Here the time scale changes: OANDA MT5 daily bars for 57 instruments
(FX majors/crosses/EM, equity indices, metals, energy, copper, natgas; datasets 85-148 minus the
7 with < 1300 bars), 2016-06 .. 2025-05. Inputs: each instrument's last 60 daily log returns;
with market=True also the cross-instrument mean of 5/20/60-day return sums and 20-day log RMS on
the same day (and the cross-sectional dispersion). One pooled model over the panel.

Weekend bars (some MT5 symbols print Sunday sessions) are dropped, so Monday's return is measured
from Friday's close; gaps up to 5 days (holidays) count as contiguous. Split: the last 20% of dates
is the test period, after a purge of obs + horizon trading days; paired bootstrap by month.

Rolling holdout (`folds`): the chronological test period (2023-07..2025-05) is one ~2-year sample.
The signals seen there (60-day return R2 0.03, direction AUC 0.535 from the instrument's own past;
market features helping 5-day volatility) are re-tested with four other 1.5-year test blocks,
purged on both sides, everything else training (splits.holdout_split).

Usage: python phase10c_daily_panel_screen.py [folds]
       (writes phase10c_daily_panel_screen.json / phase10c_daily_panel_folds.json)
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pandas as pd

from model_core.analysis import assess_target
from probe_representations import _artifact_store, _dataset_artifact_path

SHORT = {"CNHJPY", "CORN", "MXNJPY", "SOYBEANS", "SUGAR", "USDDKK", "WHEAT"}
SINCE = "2016-06-01"
CONFIGS = [("future_return", h) for h in (5, 20, 60)] + [("direction", h) for h in (5, 20, 60)] + \
          [("future_log_rv", 5), ("future_log_rv", 20), ("extreme", 20)]
TARGET_KWARGS = {"extreme": {"q": 0.99}}      # daily: the top 1% of |r| per instrument


async def _datasets():
    import database
    from sqlalchemy import select
    from data.models import Dataset
    async with database.async_session_factory() as db:
        rows = (await db.execute(select(Dataset).where(Dataset.id.between(85, 148)))).scalars().all()
        return [(d.id, d.symbol, d.artifact_path) for d in rows]


def load_panel() -> dict[str, pd.Series]:
    out = {}
    for _, sym, path in sorted(asyncio.run(_datasets())):
        if sym in SHORT:
            continue
        c = pd.read_parquet(_artifact_store() / path, columns=["close"])["close"].sort_index()
        c = c[(c.index >= SINCE) & (c.index.dayofweek < 5)]
        out[sym] = c[~c.index.duplicated(keep="last")]
    return out


FOLDS = [("2018-01-01", "2019-07-01"), ("2019-07-01", "2021-01-01"), ("2021-01-01", "2022-07-01"),
         ("2022-07-01", "2024-01-01")]
FOLD_CONFIGS = [("future_return", 20), ("future_return", 60), ("direction", 20), ("direction", 60),
                ("future_log_rv", 5)]


def folds():
    out_path = Path("phase10c_daily_panel_folds.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    panel = load_panel()
    for start, end in FOLDS:
        for target, h in FOLD_CONFIGS:
            key = f"{target}_h{h}_{start[:7]}"
            if key in results:
                continue
            r = assess_target(panel, target=target, horizon=h, obs=60, with_time=False, seed=0,
                              bootstrap_group="month", max_gap="5D", market=True, test_period=(start, end))
            results[key] = r
            out_path.write_text(json.dumps(results, indent=1))
            m = r["metrics"]
            k = "auc" if r["task"] == "classification" else "r2"
            print(f"{key:28s} own {m['linear_own'][k]:+.4f} lin {m['linear'][k]:+.4f} "
                  f"hgb {m['hgb'][k]:+.4f} mlp {m['mlp'][k]:+.4f} mkt_gain {m['linear']['exog_gain_rel']*100:+.2f}% "
                  f"ci_ref {[round(x, 6) for x in m['linear']['ci95_vs_reference']]}", flush=True)


def main():
    out_path = Path("phase10c_daily_panel_screen.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    panel = load_panel()
    print(f"{len(panel)} instruments, {sum(map(len, panel.values()))} daily bars", flush=True)
    for target, h in CONFIGS:
        key = f"{target}_h{h}"
        if key in results:
            continue
        t0 = time.time()
        r = assess_target(panel, target=target, horizon=h, obs=60, with_time=False, seed=0,
                          split="chronological", bootstrap_group="month", max_gap="5D", market=True,
                          target_kwargs=TARGET_KWARGS.get(target))
        results[key] = r
        out_path.write_text(json.dumps(results, indent=1))
        m = r["metrics"]
        k = "auc" if r["task"] == "classification" else "r2"
        best = max((n for n in m if n not in ("base_rate", "persistence")), key=lambda n: m[n][k])
        print(f"{key:18s} {r['verdict']:18s} n={r['n_train']}/{r['n_test']} own {m['linear_own'][k]:+.4f} "
              f"lin {m['linear'][k]:+.4f} best {best} {m[best][k]:+.4f} "
              f"mkt_gain {m['linear']['exog_gain_rel']*100:+.2f}% ci {[round(x, 5) for x in m['linear']['exog_ci95']]} "
              f"({time.time() - t0:.0f}s) -- {r['reason']}", flush=True)


if __name__ == "__main__":
    import sys
    folds() if sys.argv[1:] == ["folds"] else main()
