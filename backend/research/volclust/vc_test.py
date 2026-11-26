"""Pre-execution checks on synthetic inputs only (no research data is read): generates a synthetic input set in VC_SYNTH and runs the unit checks.
The synthetic HistData series are stamped in New York local time (some weeks opening at 16:00), the synthetic main series on New York + 7 h.
usage: VC_SYNTH=<dir> python vc_test.py   then   VC_SYNTH=<dir> VC_B=40 VC_BCAL=40 VC_NCAL=20 python vc_run.py"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

import vc_env as E
import vc_core as C
import vc_data as D
import dr_env as DE
import dr_b_windows as BW
import dr_b_rules as BR
from dr_a1_stats import IDX, STAT_NAMES

assert E.SYNTH, "set VC_SYNTH"
BW.atr_groups = lambda: [0.005, 0.01, 0.015]


def series(start, end, seed):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, end, freq="1min")
    s = t + pd.Timedelta(hours=7)                       # trading week: Sunday 17:00 -> Friday 17:00
    t = t[s.dayofweek < 5]
    tod = (t.hour * 60 + t.minute).values / 1440
    n = len(t)
    h = np.zeros(n // 200 + 1)
    for i in range(1, len(h)):
        h[i] = 0.9 * h[i - 1] + 0.3 * rng.standard_normal()
    vol = 1e-4 * (1.0 + 0.8 * np.sin(2 * np.pi * tod) ** 2) * np.exp(0.3 * np.repeat(h, 200)[:n])
    r = vol * rng.standard_t(5, n)
    return pd.DataFrame({"close": np.round(100 * np.exp(np.cumsum(r)), 4)}, index=t)


def to_hist(local):
    """local: New York local stamps (week opens Sunday 17:00) -> (HistData-like frame with three weeks of two springs stamped 1 h early, the same rows on New York + 7 h)."""
    conv = local.copy()
    conv.index = local.index + pd.Timedelta(hours=7)
    early = np.zeros(len(local), bool)
    for a, b in (("2019-03-10", "2019-03-31"), ("2020-03-08", "2020-03-29")):
        early |= (local.index >= a) & (local.index < b)
    raw = local.copy()
    raw.index = local.index - pd.to_timedelta(early.astype(int), unit="h")
    return raw, conv


def make():
    os.makedirs(E.SYNTH, exist_ok=True)
    os.makedirs(E.DR_RESULTS, exist_ok=True)
    long = series("2009-12-01", "2021-09-10", 1)
    raw, conv = to_hist(long)
    raw.to_parquet(os.path.join(E.SYNTH, "hist_USDJPY.parquet"))
    main = conv[(conv.index >= "2016-08-31") & (conv.index <= "2021-09-01")]
    main.to_parquet(os.path.join(E.SYNTH, "main.parquet"))
    for k, sym in enumerate(E.PAIRS):
        to_hist(series("2016-08-28", "2020-09-03", 10 + k))[0].to_parquet(os.path.join(E.SYNTH, f"hist_{sym}.parquet"))
    os.makedirs(E.PREVIOUS, exist_ok=True)
    json.dump({"small_category_enabled": False}, open(os.path.join(E.PREVIOUS, "v0c.json"), "w"))
    W = BW.build(main, DE.L_MAIN, 1)
    W.to_csv(os.path.join(E.DR_RESULTS, "b_windows_L1000_bar1.csv"), index=False)
    WA, WB = BR.split_est(W)
    json.dump({"primary": BR.freeze(WA, WB, DE.M_NAMES, DE.E_NAMES)}, open(os.path.join(E.DR_RESULTS, "frozen_rules.json"), "w"))
    rng = np.random.default_rng(3)
    meta, runs, pts = [], {}, []
    for pid, d, w in DE.grid():
        fail = rng.random() < 0.2
        for s in ("SA", "SB"):
            for k in range(5):
                good = not (fail and rng.random() < 0.3)
                meta.append({"set": s, "pid": pid, "k": k, "success": bool(good)})
                if good:
                    t = D.ticks(s, pid, k)
                    res = {"win": {}}
                    for tpc in DE.TPCS:
                        c = t[::tpc] * DE.TICK
                        R = np.array([np.diff(np.log(x)) for x in [c[i * 1000:i * 1000 + 1001] for i in range((len(c) - 1) // 1000)]])
                        a = np.zeros((len(R), len(STAT_NAMES)))
                        a[:, IDX["volclust_50"]] = C.volclust(R)
                        res["win"][tpc] = a
                    runs[(s, pid, k)] = res
        row = {"pid": pid, "delta": d, "wma": w, "all10": all(m["success"] for m in meta[-10:])}
        base = {"acf_1": rng.normal(0, 0.04), "kurtosis": rng.uniform(4, 15), "volclust_50": rng.normal(0.03, 0.03), "Q": rng.uniform(3.5, 6)}
        for s in ("SA", "SB"):
            for m, se in (("acf_1", 0.004), ("kurtosis", 0.4), ("volclust_50", 0.003), ("Q", 0.05)):
                row[f"{s}_{m}"], row[f"{s}_{m}_se"] = base[m] + rng.normal(0, se), se
        pts.append(row)
    pickle.dump({"runs": runs}, open(os.path.join(E.SYNTH, "a1_stats.pkl"), "wb"))
    open(os.path.join(E.SYNTH, "meta.jsonl"), "w").write("\n".join(json.dumps(m) for m in meta))
    pd.DataFrame(pts).to_csv(os.path.join(E.DR_RESULTS, "a1_points.csv"), index=False)
    return main, W


def checks(main, W):
    rng = np.random.default_rng(5)
    # 1. vectorised volclust equals the atlas definition
    worst = 0.0
    for _ in range(20):
        c = 100 * np.exp(np.cumsum(1e-4 * rng.standard_t(3, 1001)))
        a = DE and __import__("ddm_regime_atlas").full_characterize(pd.DataFrame({"close": c}))["volclust_50"]
        worst = max(worst, abs(a - C.volclust(np.diff(np.log(c))[None, :])[0]))
    assert worst <= 1e-12, worst
    # 2. window builder equals the previous study's builder (with a few removed blocks to create gaps)
    df = main.iloc[:400_000].copy()
    drop = np.zeros(len(df), bool)
    for i in rng.integers(0, len(df) - 20, 60):
        drop[i:i + rng.integers(1, 12)] = True
    df = df[~drop]
    old = BW.build(df, 1000, 1)
    old = old[old["period"] == "estimation"]
    new = C.build_windows(C.prepare(df), 1000, E.ESTIM)
    assert np.array_equal(old["row"].values, new["row"]), (len(old), len(new["row"]))
    assert np.array_equal(old["excluded"].values, new["excluded"]) and np.array_equal(old["week"].values, new["week"])
    # 3. leave-week-out profile equals brute force (with merging of sparse bins)
    S = C.prepare(df.iloc[:60_000])
    per = ("2016-08-31", "2016-12-31")
    win, pool = C.keep(C.build_windows(S, 1000, per)), C.pool_returns(S, per)
    se = C.Season(pool, win, "ny15")
    m = np.bincount(rng.integers(0, len(se.units), len(se.units)), minlength=len(se.units)).astype(float)
    P = se.profiles(m)
    pb, nb = C.bins_of(pool["ts"], "ny15")
    for u in range(len(se.units)):
        w = np.array([0.0 if v == u else m[v] for v in range(len(se.units))])[np.searchsorted(se.units, pool["week"])]
        s = np.bincount(pb, weights=w * np.abs(pool["r"]), minlength=nb)
        n = np.bincount(pb, weights=w, minlength=nb)
        groups, cur = [], []
        for b in range(nb):
            if n[b] >= E.MIN_BIN_OBS:
                if any(n[x] >= E.MIN_BIN_OBS for x in cur):
                    groups.append(cur)
                    cur = []
            cur.append(b)
        groups.append(cur)
        for g in groups:
            if n[g].sum() > 0:
                assert np.allclose(P[u, g], s[g].sum() / n[g].sum(), rtol=1e-10), (u, g[:3])
    assert np.isfinite(se.adjusted(win["R"], np.ones(len(se.units)))).all()
    # 4. basis New York + 7 h -> New York bins; the fixed UTC-5 sensitivity differs by 1 h in summer only
    t = np.array([np.datetime64("2015-07-01T12:00:00"), np.datetime64("2015-01-05T12:00:00")]).astype("datetime64[s]").astype("int64")
    assert list(C.bins_of(t, "ny15")[0]) == [20, 20] and list(C.bins_of(t, "est15")[0]) == [16, 20]
    assert list(C.bins_of(t, "ny_wd30")[0]) == [2 * 48 + 10, 10]                       # Wednesday and Monday 05:00 New York
    # 4b. the HistData conversion recovers the main-series stamps, including the weeks stamped 1 h early
    got, info = C.convert_hist(pd.read_parquet(os.path.join(E.SYNTH, "hist_USDJPY.parquet")))
    both = got.index.intersection(main.index)
    assert len(both) >= len(main) - 5 and np.array_equal(got.loc[both, "close"].values, main.loc[both, "close"].values), (len(both), len(main))
    assert len(info["weeks_shifted_8h"]) == 6, info
    # 5. classification and DDM bootstrap
    assert C.classify([[0.03, 0.04]] * 4, 0.022, True) == "差あり" and C.classify([[0.0, 0.01]] * 4, 0.022, True) == "差は小さい"
    assert C.classify([[0.0, 0.01]] * 4, 0.022, False) == "判定保留" and C.classify([[0.03, 0.04]] * 3 + [[0.01, 0.04]], 0.022, True) == "判定保留"
    X, n = C.pad_runs([np.array([1.0, 1, 1]), np.array([2.0, 2, 2, 2, 2])])
    assert np.allclose(C.ddm_max_boot(X, n, 10, rng), 2.0) and np.allclose(C.point_means(X, n), [1, 2])
    print("unit checks passed", flush=True)


if __name__ == "__main__":
    checks(*make())
