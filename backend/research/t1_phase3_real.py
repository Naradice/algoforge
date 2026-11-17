"""T1 Phase 3 (Study Manager #895, plan v0.4 S9): distance of the model clocks to REAL quote-update statistics.

Real data = MT5 USDJPY bid/ask quote updates (millisecond stamps, broker time). These are quote updates, NOT executions.
Pre-registered design (fixed before looking at any model-vs-real result). AMENDMENT made before any real-data
statistic was computed: the tick file has no data between 2021-11-29 and 2022-02-28, so the planned periods
(2021-09..11 / 2021-12..2022-02) were shifted to 2021-06..08 (selection) and 2021-09-01..2021-11-26 (evaluation).
  * selection window: the 5 consecutive broker-time hours (no wrap) with the highest mean distinct-timestamp update
    rate on weekdays of 2021-06-01..2021-08-31; this period is NOT used for evaluation.
  * evaluation: weekdays 2021-09-01..2021-11-26, days with >= 500 updates in the window; update = distinct timestamp.
  * statistics (scale-free only): ACF(log dt,1), CV(dt), ACF(updates per M1 bar, 1), ACF(|M1 mid return|, 1) on the
    continuous grid inside the window (mid = (bid+ask)/2, forward-filled).
  * 'C1 improved' only if, for >= 3 of the 4 statistics, C1 is closer to the real value than BOTH C0s and C2 with the
    95% bootstrap CI (real days x model price seeds) of each distance difference excluding 0.
  * M2-S2 parameters are not re-selected on real data.

Usage: python research/t1_phase3_real.py <full_result.json> <out.json> [tick_file]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

TICK_FILE = r"L:\data\fx\USDJPY_201112190000_202202280529.csv"
PRE = ("2021-06-01", "2021-08-31")
EVAL = ("2021-09-01", "2021-11-26")
WINDOW_HOURS = 5
MIN_UPDATES_PER_DAY = 500
N_BOOT = 10_000


def find_offset(f, size, target):
    """Byte offset of the first line whose date >= target (YYYY.MM.DD) by bisection on the sorted file."""
    lo, hi = 0, size
    while hi - lo > 1 << 16:
        mid = (lo + hi) // 2
        f.seek(mid)
        f.readline()
        line = f.readline().decode("ascii", "ignore")
        d = line.split("\t")[0]
        if not d or d[:4] < "2000":
            hi = mid
            continue
        if d < target:
            lo = mid
        else:
            hi = mid
    f.seek(lo)
    f.readline()
    while True:
        pos = f.tell()
        d = f.readline().decode("ascii", "ignore").split("\t")[0]
        if not d or d >= target:
            return pos


def read_range(path, start, end, chunk=4_000_000):
    """Yield DataFrames (ts_ns, bid, ask) for rows with start <= date <= end (dates as 'YYYY.MM.DD')."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        off = find_offset(f, size, start)
        f.seek(off)
        for df in pd.read_csv(f, sep="\t", header=None, names=["d", "t", "bid", "ask", "last", "vol", "flags"],
                              usecols=[0, 1, 2, 3], dtype={"d": str, "t": str, "bid": float, "ask": float},
                              chunksize=chunk):
            df = df[df["d"] <= end]
            if df.empty:
                break
            ts = pd.to_datetime(df["d"] + " " + df["t"], format="%Y.%m.%d %H:%M:%S.%f").values.astype("datetime64[ns]").astype("int64")
            yield pd.DataFrame({"ts": ts, "bid": df["bid"].values, "ask": df["ask"].values})
            if df["d"].iloc[-1] >= end and len(df) < chunk:
                break


def select_window(path):
    counts, days = {}, {}
    for df in read_range(path, PRE[0].replace("-", "."), PRE[1].replace("-", ".")):
        t = pd.to_datetime(df["ts"].values)
        keep = t.dayofweek < 5
        t = t[keep]
        ts = df["ts"].values[keep]
        # distinct timestamps per (date, hour)
        key = pd.DataFrame({"ts": ts, "date": t.date, "hour": t.hour}).drop_duplicates("ts")
        g = key.groupby(["date", "hour"]).size()
        for (d, h), n in g.items():
            counts[(d, h)] = counts.get((d, h), 0) + n
    dates = sorted({d for d, _ in counts})
    rate = np.zeros(24)
    for h in range(24):
        rate[h] = np.mean([counts.get((d, h), 0) for d in dates])
    best = max(range(24 - WINDOW_HOURS + 1), key=lambda h: rate[h:h + WINDOW_HOURS].sum())
    return best, rate.tolist(), len(dates)


def acf1(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    d = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / d) if d > 0 else float("nan")


def day_stats(ts_ns, bid, ask):
    """ts in ns (sorted), bid/ask with NaN for the unchanged side."""
    keep = np.r_[True, np.diff(ts_ns) > 0]              # one update per distinct timestamp (last row wins below)
    # forward fill both sides, then take the last row of each distinct timestamp
    b = pd.Series(bid).ffill().values
    a = pd.Series(ask).ffill().values
    last_of_ts = np.r_[np.diff(ts_ns) > 0, True]
    ts, b, a = ts_ns[last_of_ts], b[last_of_ts], a[last_of_ts]
    ok = ~(np.isnan(b) | np.isnan(a))
    ts, b, a = ts[ok], b[ok], a[ok]
    if len(ts) < MIN_UPDATES_PER_DAY:
        return None
    dt = np.diff(ts) / 1e9
    mid = (b + a) / 2
    t0 = ts[0] // 60_000_000_000 * 60_000_000_000
    bins = ((ts - t0) // 60_000_000_000).astype(np.int64)
    nb = int(bins.max()) + 1
    counts = np.bincount(bins, minlength=nb).astype(float)
    ne = np.flatnonzero(counts)
    first_idx = np.searchsorted(bins, ne, "left")
    last_idx = np.append(first_idx[1:], len(bins)) - 1
    close = np.full(nb, np.nan)
    close[ne] = mid[last_idx]
    close = pd.Series(close).ffill().values
    r = np.diff(np.log(close))
    a_r = np.abs(r)
    return {"acf_logdt1": acf1(np.log(dt)), "cv_dt": float(dt.std() / dt.mean()), "count_acf1": acf1(counts),
            "absret_acf1": acf1(a_r) if a_r.std() > 0 else float("nan"), "n_updates": int(len(ts)),
            "mean_dt": float(dt.mean()), "zero_ret_rate": float(np.mean(r == 0)), "n_bars": nb}


def eval_days(path, h0):
    per_day = {}
    buf = {}
    cur = None
    for df in read_range(path, EVAL[0].replace("-", "."), EVAL[1].replace("-", ".")):
        t = pd.to_datetime(df["ts"].values)
        m = (t.dayofweek < 5) & (t.hour >= h0) & (t.hour < h0 + WINDOW_HOURS)
        if not m.any():
            continue
        sub = df[m]
        dates = pd.to_datetime(sub["ts"].values).date
        for d in np.unique(dates):
            sel = dates == d
            buf.setdefault(d, []).append(sub[sel])
    for d, parts in buf.items():
        x = pd.concat(parts)
        s = day_stats(x["ts"].values, x["bid"].values, x["ask"].values)
        if s is not None:
            per_day[str(d)] = s
    return per_day


def boot_distance(real_days, model_vals, stat_real, rng):
    """Bootstrap over real days and model seeds jointly: returns draws of mean(real) and arrays for models."""
    out = []
    for _ in range(N_BOOT):
        r = np.mean(rng.choice(real_days, size=len(real_days)))
        out.append((r, {c: np.mean(rng.choice(v, size=len(v))) for c, v in model_vals.items()}))
    return out


def main():
    result_json, out_json = sys.argv[1], sys.argv[2]
    path = sys.argv[3] if len(sys.argv) > 3 else TICK_FILE
    full = json.load(open(result_json))
    per_seed = full["per_seed"]
    h0, rate, ndays = select_window(path)
    print(f"selected window broker hours {h0}-{h0 + WINDOW_HOURS} (pre-period weekdays: {ndays}); "
          f"mean updates/hour in the window: {sum(rate[h0:h0 + WINDOW_HOURS]) / WINDOW_HOURS:.0f}")
    days = eval_days(path, h0)
    print(len(days), "evaluation days")
    stat_map = {
        "acf_logdt1": ("acf_logdt1", lambda r, c: r["clock"][c]["acf_logT1"]),
        "cv_dt": ("cv_dt", lambda r, c: r["clock"][c]["cv"]),
        "count_acf1": ("count_acf1", lambda r, c: r["grid"][c]["cont"]["count_acf1"]),
        "absret_acf1": ("absret_acf1", lambda r, c: r["grid"][c]["cont"]["absacf1"]),
    }
    rng = np.random.default_rng(31415)
    res = {"window_hours": [h0, h0 + WINDOW_HOURS], "hourly_rate": rate, "n_days": len(days),
           "real_mean": {}, "model_mean": {}, "improved": {}}
    cnt = 0
    for name, (key, getter) in stat_map.items():
        real = np.array([d[key] for d in days.values() if np.isfinite(d[key])])
        model = {c: np.array([getter(r, c) for r in per_seed]) for c in ("C0", "C0s", "C1", "C2", "C3")}
        draws = boot_distance(real, model, None, rng)
        d_c0s = np.array([abs(m["C1"] - r) - abs(m["C0s"] - r) for r, m in draws])
        d_c2 = np.array([abs(m["C1"] - r) - abs(m["C2"] - r) for r, m in draws])
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        res["real_mean"][name] = float(real.mean())
        res["model_mean"][name] = {c: float(v.mean()) for c, v in model.items()}
        better = bool(ci(d_c0s)[1] < 0 and ci(d_c2)[1] < 0)
        cnt += better
        res["improved"][name] = {"better_than_C0s_and_C2": better, "dist_diff_C1_minus_C0s": float(d_c0s.mean()), "ci_C0s": ci(d_c0s),
                                 "dist_diff_C1_minus_C2": float(d_c2.mean()), "ci_C2": ci(d_c2)}
        print(f"{name}: real {real.mean():.4f}  model " + "  ".join(f"{c} {v.mean():.4f}" for c, v in model.items()) +
              f" | C1-vs-C0s dist diff {d_c0s.mean():+.4f} {ci(d_c0s)}  C1-vs-C2 {d_c2.mean():+.4f} {ci(d_c2)}  -> {better}")
    res["n_stats_improved"] = cnt
    res["conclusion"] = "C1 improved (>=3 of 4 pre-specified statistics)" if cnt >= 3 else "no improvement claim (<3 of 4)"
    res["real_other"] = {"mean_dt_s": float(np.mean([d["mean_dt"] for d in days.values()])),
                         "zero_ret_rate": float(np.mean([d["zero_ret_rate"] for d in days.values()]))}
    print("conclusion:", res["conclusion"])
    json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
