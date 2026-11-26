"""Loaders of the research inputs (real market series, stored DDM results). In synthetic mode (VC_SYNTH) the same interfaces return generated data."""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

import vc_env as E
import vc_core as C
import dr_env as DE
import dr_b_windows as BW
from dr_a1_stats import IDX


def load_main():
    """the main series of the previous study (final period removed by its loader)."""
    if E.SYNTH:
        return pd.read_parquet(os.path.join(E.SYNTH, "main.parquet"))
    df, _ = BW.load_raw()
    return df


def load_hist(sym, end):
    """HistData 1-minute closes converted to the basis of the main series (New York + 7 h, vc_core.convert_hist) up to the end date (+1 day, converted basis).
    The unconverted rows are cached. Duplicate time stamps: the first row is kept."""
    limit = pd.Timestamp(end) + pd.Timedelta(days=3)
    if E.SYNTH:
        raw = pd.read_parquet(os.path.join(E.SYNTH, f"hist_{sym}.parquet"))
        raw = raw[raw.index < limit]
        info = {"duplicates_dropped": 0}
    else:
        path = os.path.join(E.HIST_DIR, f"HistData_{sym}_1MIN.csv")
        cache = os.path.join(E.OUT, f"hist_raw_{sym}_{end}.parquet")
        info = {"source_file": path, "source_bytes": os.path.getsize(path)}
        if os.path.exists(cache):
            raw = pd.read_parquet(cache)
            info.update(json.load(open(cache + ".json")))
        else:
            raw = pd.read_csv(path, usecols=["time", "close"])
            raw["time"] = pd.to_datetime(raw["time"], format="%Y-%m-%d %H:%M:%S")
            raw = raw.set_index("time").sort_index(kind="stable")
            raw = raw[raw.index < limit]
            dup = raw.index.duplicated(keep="first")
            raw = raw[~dup]
            extra = {"duplicates_dropped": int(dup.sum()), "sha256": BW.sha256_file(path)}
            raw.to_parquet(cache)
            json.dump(extra, open(cache + ".json", "w"))
            info.update(extra)
    df, conv = C.convert_hist(raw)
    df = df[df.index < pd.Timestamp(end) + pd.Timedelta(days=1)]
    cut = str((pd.Timestamp(end) + pd.Timedelta(days=1)).date())
    conv["weeks_shifted_8h"] = [w for w in conv["weeks_shifted_8h"] if w < cut]
    info.update(conv)
    return df, info


def b_windows():
    p = os.path.join(E.DR_RESULTS, "b_windows_L1000_bar1.csv")
    return pd.read_csv(p)


def dr_json(name):
    return json.load(open(os.path.join(E.DR_RESULTS, name), encoding="utf-8"))


def a1_points():
    return pd.read_csv(os.path.join(E.DR_RESULTS, "a1_points.csv"))


_A1 = {}


def a1():
    if not _A1:
        if E.SYNTH:
            _A1["runs"] = pickle.load(open(os.path.join(E.SYNTH, "a1_stats.pkl"), "rb"))["runs"]
            _A1["meta"] = [json.loads(x) for x in open(os.path.join(E.SYNTH, "meta.jsonl"))]
        else:
            _A1["runs"] = pickle.load(open(E.A1_STATS, "rb"))["runs"]
            _A1["meta"] = [json.loads(x) for x in open(os.path.join(DE.A1_DIR, "meta.jsonl"), encoding="utf-8")]
    return _A1


def ticks(s, pid, k):
    if E.SYNTH:
        rng = np.random.default_rng(DE.seed_of(s, pid, k))
        return (100_000 + np.cumsum(rng.integers(-3, 4, DE.N_TRADES))).astype(np.int32)
    return np.load(DE.run_file(s, pid, k))


def ddm_points():
    """base = points with all 10 runs successful; extended = base + partially failed points with >= 3 successful runs in each set."""
    ok = {}
    for m in a1()["meta"]:
        if m["success"]:
            ok.setdefault((m["pid"], m["set"]), []).append(m["k"])
    base, ext = [], []
    for pid, _, _ in DE.grid():
        n = [len(ok.get((pid, s), [])) for s in ("SA", "SB")]
        if min(n) == DE.SEEDS_PER_SET:
            base.append(pid)
        if min(n) >= 3:
            ext.append(pid)
    return base, ext, ok


def ddm_run_values(pids, ok, tpc=DE.TPC_MAIN):
    """{set: list per point of arrays (one value per successful run = mean volclust_50 over the run's windows)} from the stored window statistics."""
    R = a1()["runs"]
    return {s: [np.array([R[(s, p, k)]["win"][tpc][:, IDX["volclust_50"]].mean() for k in sorted(ok[(p, s)])]) for p in pids] for s in ("SA", "SB")}


def padded(vals):
    return {s: C.pad_runs(vals[s]) for s in ("SA", "SB")}


def ddm_returns(s, pid, k, tpc, L):
    """[windows, L] returns of one stored run re-aggregated at tpc (L=4000: the first window only)."""
    t = ticks(s, pid, k)[::tpc].astype(np.float64) * DE.TICK
    n = (len(t) - 1) // L
    if L == E.L_SENS:
        n = min(n, 1)
    return np.array([np.diff(np.log(t[i * L:i * L + L + 1])) for i in range(n)])


def main_windows(period, drop_excluded=True):
    S = C.prepare(load_main())
    W = C.build_windows(S, E.L_MAIN, period)
    return S, (C.keep(W) if drop_excluded else W)
