"""P3 audit measurement (NOT an experiment; no realised data, no scoring): instrumentation of the F1 DDM path generator
(f1_ddm.simulate_long_path settings) -> steps per trade, early-vs-late candle volatility, shock-event timing.
Parameters: one representative candidate (delta=0, wma=7), seeds 0..11, N_AGENT=300, SHOCK as in f1_ddm.py."""
import json, os, random, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from data.collectors.ddm_simulator import DDMv3
SHOCK = dict(exogenous_shock_probability=0.002, exogenous_shock_size=0.3, exogenous_shock_decay_tau=300.0, exogenous_shock_floor_probability=0.0)
out = {"params": {"delta": 0.0, "wma": 7, "seeds": list(range(12))}, "paths": []}
for seed in range(12):
    random.seed(seed); np.random.seed(seed)
    m = DDMv3(num_agent=300, max_volatility=0.02, min_volatility=0.01, wma=7, dealer_sensitive_min=-3.5, dealer_sensitive_max=-1.5,
              spread_feedback_a=None, spread_feedback_window=150, **SHOCK)
    n_tr = 1441 * 20
    p = m.simulate(n_trades=n_tr)["price"].values
    c = p.reshape(-1, 20)[:, -1]
    r = np.diff(np.log(c))
    bins = [(0, 5), (5, 15), (15, 60), (60, 360), (360, 1440)]
    steps = [e[0] for e in m._shock_events]
    out["paths"].append({"seed": seed, "steps_per_trade": m._step_count / n_tr, "mean_abs_ret_by_bin": [float(np.abs(r[a:b]).mean()) for a, b in bins],
                         "n_shock_events": len(steps), "shock_event_steps": steps})
P = out["paths"]
out["summary"] = {"steps_per_trade_mean": float(np.mean([x["steps_per_trade"] for x in P])),
                  "mean_abs_ret_by_bin_ratio_to_last": (np.mean([x["mean_abs_ret_by_bin"] for x in P], axis=0) / np.mean([x["mean_abs_ret_by_bin"][-1] for x in P])).tolist(),
                  "mean_shock_events_per_path": float(np.mean([x["n_shock_events"] for x in P])),
                  "bins_candle_index": [[0, 5], [5, 15], [15, 60], [60, 360], [360, 1440]]}
json.dump(out, open(os.path.join(HERE, "audit_measure_results.json"), "w"), indent=1)
print(out["summary"])
