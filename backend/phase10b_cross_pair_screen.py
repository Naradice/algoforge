"""
Phase 10B: do other currency pairs carry information about USDJPY that its own past does not?

Phase 10A: on USDJPY's own last 60 M1 returns no target (return, direction, extremes, trend regime)
leaves room for a nonlinear model over a strong linear one. Here the models also see the other 8
HistData pairs (EURUSD GBPUSD AUDUSD USDCHF EURJPY GBPJPY AUDJPY CHFJPY; same source and clock):
their signed return sums (1..60 bars) and log RMS (5/20/60) up to the anchor. Timestamps are
aligned strictly (ffill_limit=0: a minute missing in any pair is a gap), so no stale quote is
carried forward.

Two questions per target, both on the same windows:
  1. linear vs linear_own     -- do the other pairs add signal at all (metrics.linear.exog_ci95)?
  2. nonlinear vs linear      -- is there headroom over the strong linear model *with* the pairs?

Caveat for h=1: bar closes are the last quote in each minute, taken at different seconds per pair,
and USDJPY ~ EURJPY / EURUSD (triangular). A one-bar "lead" from the crosses can be mostly
non-synchronous closing, not economics.

Usage: python phase10b_cross_pair_screen.py [ffill_limit]   (writes phase10b_cross_pair_screen[_ffN].json)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from model_core.analysis import align_closes, assess_target
from phase10b_data import load_closes

CONFIGS = [("future_return", h) for h in (1, 5, 20, 60)] + [("direction", h) for h in (1, 5, 20, 60)] + \
          [("future_log_rv", 20), ("future_log_rv", 60), ("vol_change", 20), ("extreme", 20)]


def main():
    ffill = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_path = Path(f"phase10b_cross_pair_screen{'' if ffill == 0 else f'_ff{ffill}'}.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    S = load_closes()
    close, exog = align_closes(S.pop("USDJPY"), S, ffill_limit=ffill)
    print(f"USDJPY + {list(exog.columns)}: {len(close)} rows {close.index[0]} .. {close.index[-1]}", flush=True)
    for target, h in CONFIGS:
        key = f"{target}_h{h}"
        if key in results:
            continue
        t0 = time.time()
        # 200k training windows (10A used 300k): the machine's commit charge is shared with other jobs.
        r = assess_target(close, target=target, horizon=h, obs=60, with_time=True, seed=0, exog=exog,
                          n_train=200_000)
        r["rows"] = [str(close.index[0]), str(close.index[-1])]
        results[key] = r
        out_path.write_text(json.dumps(results, indent=1))
        m = r["metrics"]
        k = "auc" if r["task"] == "classification" else "r2"
        best = max((n for n in m if n not in ("base_rate", "persistence")), key=lambda n: m[n][k])
        print(f"{key:18s} {r['verdict']:18s} own {m['linear_own'][k]:+.4f} lin {m['linear'][k]:+.4f} "
              f"best {best} {m[best][k]:+.4f} exog_gain {m['linear']['exog_gain_rel']*100:+.2f}% "
              f"ci {m['linear']['exog_ci95']} ({time.time() - t0:.0f}s) -- {r['reason']}", flush=True)


if __name__ == "__main__":
    main()
