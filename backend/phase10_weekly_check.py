"""
Phase 10D/E follow-up: is the trees' +7-10% on activity forecasts (future mean log volume / log tick
count over the next 20 bars) just a weekly seasonal profile the linear baseline could not express?
Same windows, one aux series, linear baseline with vs without weekday x hour dummies (and their
products with the 20-bar aux mean) -- assess_target(weekly=True).

Usage: python phase10_weekly_check.py [volume_158|ticks_165]   (one case per process: the DB
engine is bound to one event loop)   (writes phase10_weekly_check.json)
"""
from __future__ import annotations

import json
from pathlib import Path

from model_core.analysis import assess_target
from phase10d_volume_screen import load as load_volume
from phase10e_microstructure_screen import load as load_ticks


def main():
    out_path = Path("phase10_weekly_check.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    cases = {"volume_158": lambda: (lambda c, a: (c, a[["log_volume"]]))(*load_volume(158)),
             "ticks_165": lambda: (lambda c, a: (c, a[["log_ticks"]]))(*load_ticks(165))}
    import sys
    for name, loader in cases.items():
        if sys.argv[1:] and name not in sys.argv[1:]:
            continue
        close, aux = loader()
        for weekly in (False, True):
            key = f"{name}_weekly{int(weekly)}"
            if key in results:
                continue
            r = assess_target(close, target="future_aux", horizon=20, obs=60, with_time=True, seed=0,
                              aux=aux, n_train=100_000, weekly=weekly)   # 100k: the wide weekly design must fit in memory
            results[key] = r
            out_path.write_text(json.dumps(results, indent=1))
            m = r["metrics"]
            print(f"{key:20s} {r['verdict']:18s} lin {m['linear']['r2']:+.4f} hgb {m['hgb']['r2']:+.4f} "
                  f"({m['hgb']['gain_rel']*100:+.1f}% ci {[round(x, 5) for x in m['hgb']['ci95']]}) "
                  f"mlp {m['mlp']['gain_rel']*100:+.1f}% knn {m['knn']['gain_rel']*100:+.1f}%", flush=True)


if __name__ == "__main__":
    main()
