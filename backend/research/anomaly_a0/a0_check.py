"""A0 (Study Manager #897) data checks. Definitions are fixed in docs/research-anomaly-a0-protocol.md (committed before this was run).
No detection score is computed. Price values are used only for (a) label counts in the ESTIMATION period and (b) the time-basis correlation of the evaluation
period against the converted HistData series. Output: results/a0_check.json"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "volclust"))
from vc_core import convert_hist  # noqa: E402  (HistData New York local time -> New York + 7 h, plan r1 2.2)

MAIN = os.path.join(REPO, "artifacts", "datasets", "derived", "usdjpy_1min_volatility.parquet")
MAIN_SHA = "c25eb5720a5552b0230fb6ce2354de1cee020e6b7ca95da46034b75f2a53c828"
HIST = r"L:\data\fx\HistData_USDJPY_1MIN.csv"
F1 = r"C:\tmp\f1_run1_db0926f"
F1_SHA = {"selection.json": "735d28f4c2865a99069a3d6e2b1613cf86f1cc3f51e526937945a68ea988d13a",
          "scores_final.npz": "e08b30c6fc6b1b69ab9a26f670b497d62381fe06a623ea2caad639f90bbfcc15",
          "scores_validation.npz": "65a3c46a70e7bdf837c29af1454022e5750016a4e02e9c4ac595187d50451046",
          "ddm_mapping.json": "962a9843353d81e626ad4188018d778056ec02e4d048fae37b6069341a01201a",
          "ddm_ens_seed1.npy": "074d973a16f6884b5e327ee6e14f5a1df7f465914dbb47d5791c04761132605a",
          "ddm_ens_seed2.npy": "7c07a073a4ab2261cf4109bcebdca9f5acf9a61b7e5aca6bf6cc36bca7a0efa2",
          "ddm_ens_seed3.npy": "73b14d992523ab0ae3374ee4b10be3dcce3103494751776d0d90ee64e2c2cdbb"}
PERIODS = {"estimation": ("2016-08-31", "2020-08-31"), "validation": ("2020-09-01", "2021-08-31"), "evaluation": ("2021-09-01", "2022-08-31")}
SUPP = ("2022-09-01", "2022-12-31")
F1_EVAL_ORIGINS = {"C-M1": 6123, "C-M4": 5943, "C-D": 4750}
BLOCK, GAP = 60, 300
CANDIDATES = {"C-S": 15, "C-M1": 60, "C-M4": 240, "C-D": 1440}
N_MIN = 100


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def bounds(period):
    return pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(days=1)


# ---------------------------------------------------------------- time-stamp checks -------------------------------------------------------
def stamp_checks(idx, period):
    a, b = bounds(period)
    x = idx[(idx >= a) & (idx < b)]
    d = np.diff(x.values.astype("datetime64[s]").astype("int64"))
    shifted = x + pd.Timedelta(hours=6)
    monday = shifted.normalize() - pd.to_timedelta(shifted.dayofweek.values, unit="D")
    m = (monday >= a) & (monday < b)
    first = pd.Series(x[m], index=monday[m]).groupby(level=0).min()
    ok = (first.dt.dayofweek.values == 0) & (first.dt.hour.values == 0) & (first.dt.minute.values < 30)
    return {"period": list(period), "rows": int(len(x)), "first": str(x.min()), "last": str(x.max()), "duplicates": int(x.duplicated().sum()),
            "backward_steps": int((d < 0).sum()), "steps_over_60s": int((d > 60).sum()), "steps_over_300s": int((d > GAP).sum()),
            "steps_over_3_days": int((d > 3 * 86400).sum()), "weeks": int(len(ok)), "weekly_open_conforming": int(ok.sum()),
            "weekly_open_share": float(ok.mean()) if len(ok) else None}


def correlation_check(main, hist, period):
    a, b = bounds(period)

    def returns(s):
        r = np.log(s).diff()
        dt = s.index.to_series().diff().dt.total_seconds()
        return r[(dt.values == 60) & np.isfinite(r.values)]

    rm = returns(main[(main.index >= a) & (main.index < b)])
    rh = returns(hist[(hist.index >= a - pd.Timedelta(days=2)) & (hist.index < b + pd.Timedelta(days=2))])
    cols = {}
    for off in (-1, 0, 1):
        s = rh.copy()
        s.index = s.index + pd.Timedelta(hours=off)
        j = pd.concat([rm.rename("m"), s.rename("h")], axis=1, join="inner")
        iso = (j.index + pd.Timedelta(hours=6)).isocalendar()
        week = iso["year"].values.astype(np.int64) * 100 + iso["week"].values.astype(np.int64)
        g = j.groupby(week)
        cols[f"n{off}"] = g.size()
        cols[f"c{off}"] = pd.Series({w: (v["m"].corr(v["h"]) if len(v) > 2 else np.nan) for w, v in g}, dtype=float)
    t = pd.DataFrame(cols)
    t = t[t["n0"] >= 500]
    c0 = np.nan_to_num(t["c0"].values.astype(float), nan=-np.inf)
    ok = np.isfinite(c0)
    for off in (-1, 1):
        other = np.where(t[f"n{off}"].fillna(0).values >= 100, t[f"c{off}"].values.astype(float), np.nan)
        ok &= ~(np.nan_to_num(other, nan=-np.inf) >= c0)
    share, med = float(ok.mean()), float(np.nanmedian(t["c0"]))
    return {"period": list(period), "n_weeks": int(len(t)), "n_offset0_best": int(ok.sum()), "share": share, "median_corr_offset0": med,
            "failing_weeks": [int(w) for w in t.index[~ok]], "passes": bool(share >= 0.98 and med >= 0.90)}


# ---------------------------------------------------------------- origins and labels ------------------------------------------------------
class Series:
    def __init__(self, idx):
        self.idx = idx
        self.n = len(idx)
        s = idx.values.astype("datetime64[s]").astype("int64")
        self.gaps = np.flatnonzero(np.diff(s) > GAP)

    def contiguous(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        ok = (a >= 0) & (b < self.n)
        out = np.zeros(a.shape, bool)
        out[ok] = (np.searchsorted(self.gaps, b[ok]) - np.searchsorted(self.gaps, a[ok])) == 0
        return out

    def origins(self, name, period):
        """origin rows valid by time stamps only (protocol section 3)."""
        a, b = bounds(period)
        H = CANDIDATES[name]
        step = 15 if name == "C-S" else BLOCK
        t = np.arange(step - 1, self.n, step)
        ok = self.contiguous(t - (step - 1), t) & self.contiguous(t, t + H)
        if name != "C-S":
            ok &= self.contiguous(t - step, t)
        end = np.minimum(t + H, self.n - 1)
        ok &= np.asarray((self.idx[t] >= a) & (self.idx[end] < b))
        return t[ok]

    def state_origins(self, period, horizon_blocks=120):
        """C-R: the last contiguous block of each file date whose next 120 contiguous blocks end inside the period.
        returns (positions into the list of contiguous blocks, that list of block end rows, day ordinal of each origin)."""
        a, b = bounds(period)
        t = np.arange(BLOCK - 1, self.n, BLOCK)
        t = t[self.contiguous(t - (BLOCK - 1), t)]
        t = t[np.asarray((self.idx[t] >= a) & (self.idx[t] < b))]
        day = self.idx[t].normalize()
        last = np.flatnonzero(np.r_[day[1:] != day[:-1], True])
        ordinal = np.arange(len(last))
        keep = last + horizon_blocks < len(t)
        return last[keep], t, ordinal[keep]


def events(pos, gap):
    """pos: sorted positions of positive origins; a new event starts when the difference to the previous positive exceeds gap."""
    if len(pos) == 0:
        return []
    cut = np.flatnonzero(np.diff(pos) > gap) + 1
    return np.split(pos, cut)


def summarise(origin_pos, V, q, gap, length):
    thr = float(np.quantile(V, q))
    flag = V > thr
    ev = events(origin_pos[flag], gap)
    dur = [int(e[-1] - e[0] + length) for e in ev]
    return {"threshold": thr, "positive_origins": int(flag.sum()), "events": int(len(ev)),
            "duration_median": float(np.median(dur)) if dur else None, "duration_max": int(max(dur)) if dur else None}


def label_counts(S, close):
    """ESTIMATION period only. S and close are truncated to the end of the estimation period by the caller."""
    r = np.r_[0.0, np.diff(np.log(close))]
    cs = np.cumsum(r * r)
    blocks = np.arange(BLOCK - 1, S.n, BLOCK)
    ok_block = S.contiguous(blocks - (BLOCK - 1), blocks)
    vb = blocks[ok_block]
    brv = cs[vb] - cs[vb - (BLOCK - 1)]                      # within-block realised variance (59 returns)
    cb = np.r_[0.0, np.cumsum(brv)]
    out = {}
    for name, H in CANDIDATES.items():
        t = S.origins(name, PERIODS["estimation"])
        if name != "C-S":                                    # F1 also requires a positive realised variance of the origin block
            t = t[(cs[t] - cs[t - BLOCK]) > 0]
        V = cs[t + H] - cs[t]
        G = max(240, H)
        res = {"H_rows": H, "valid_origins": int(len(t)), "primary_q95": summarise(t, V, 0.95, G, H), "q99": summarise(t, V, 0.99, G, H),
               "q95_gap_max60H": summarise(t, V, 0.95, max(60, H), H), "q95_gap_max1440H": summarise(t, V, 0.95, max(1440, H), H)}
        k = np.searchsorted(vb, t, side="right")             # number of contiguous blocks ending at or before the origin
        have = k >= 24
        past = (cb[k[have]] - cb[k[have] - 24]) / 24
        nz = past > 0
        res["normalised_q95"] = summarise(t[have][nz], V[have][nz] / (past[nz] * H / BLOCK), 0.95, G, H)
        res["normalised_q95"]["valid_origins"] = int(nz.sum())
        out[name] = res
    pos, tb, day = S.state_origins(PERIODS["estimation"])
    rv = cs[tb] - cs[tb - (BLOCK - 1)]
    c = np.r_[0.0, np.cumsum(rv)]
    V = (c[pos + 121] - c[pos + 1]) / 120
    span = 120
    res = {"H_blocks": 120, "valid_origins": int(len(pos))}
    for key, q in (("primary_q95", 0.95), ("q99", 0.99)):
        s = summarise(day, V, q, 5, 0)
        res[key] = {k: v for k, v in s.items() if not k.startswith("duration")}
    have = pos + 1 >= 24
    past = (c[pos[have] + 1] - c[pos[have] - 23]) / 24
    s = summarise(day[have], V[have] / past, 0.95, 5, 0)
    res["normalised_q95"] = {**{k: v for k, v in s.items() if not k.startswith("duration")}, "valid_origins": int(have.sum())}
    res["horizon_blocks"] = span
    out["C-R"] = res
    return out


def main():
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "backend/research/anomaly_a0/a0_check.py", "docs/research-anomaly-a0-protocol.md",
                                     "backend/research/volclust/vc_core.py"], text=True).strip()
    if dirty:
        raise SystemExit("uncommitted: " + dirty)
    t0 = time.time()
    out = {"_code": {"head": head}, "inputs": {"main": {"path": MAIN, "sha256": sha(MAIN)}}}
    assert out["inputs"]["main"]["sha256"] == MAIN_SHA
    df = pd.read_parquet(MAIN, columns=["close"])
    idx = pd.DatetimeIndex(df.index)
    out["stamps_main"] = {k: stamp_checks(idx, p) for k, p in PERIODS.items()}
    # HistData: time basis of the evaluation period, time stamps of the supplementary period
    raw = pd.read_csv(HIST, usecols=["time", "close"])
    raw["time"] = pd.to_datetime(raw["time"], format="%Y-%m-%d %H:%M:%S")
    raw = raw[raw["time"] >= "2021-08-20"].set_index("time").sort_index(kind="stable")
    n_dup = int(raw.index.duplicated().sum())
    hist, conv = convert_hist(raw[~raw.index.duplicated(keep="first")])
    out["inputs"]["histdata"] = {"path": HIST, "sha256": sha(HIST), "duplicates_from_2021_08_20": n_dup, "conversion": conv}
    out["time_basis_evaluation"] = correlation_check(df["close"], hist["close"], PERIODS["evaluation"])
    out["stamps_hist_supplementary"] = stamp_checks(hist.index, SUPP)
    # origins by time stamps only
    S = Series(idx)
    counts = {}
    for per in ("validation", "evaluation"):
        counts[per] = {name: int(len(S.origins(name, PERIODS[per]))) for name in CANDIDATES}
        counts[per]["C-R"] = int(len(S.state_origins(PERIODS[per])[0]))
    out["origins_by_stamps"] = counts
    out["f1_recorded_evaluation_origins"] = F1_EVAL_ORIGINS
    # label counts: estimation period only (arrays truncated before any value is used)
    end = int(np.searchsorted(idx.values, np.datetime64(bounds(PERIODS["estimation"])[1])))
    out["labels_estimation"] = label_counts(Series(idx[:end]), df["close"].values[:end].astype(float))
    out["labels_estimation_rows_used"] = {"rows": end, "last": str(idx[end - 1])}
    del df
    # feasibility rule M3
    m3 = {}
    for name, res in out["labels_estimation"].items():
        n_eval = F1_EVAL_ORIGINS.get(name, counts["evaluation"][name])
        expected = res["primary_q95"]["events"] / res["valid_origins"] * n_eval
        m3[name] = {"events_estimation": res["primary_q95"]["events"], "origins_estimation": res["valid_origins"], "origins_evaluation": n_eval,
                    "origins_evaluation_source": "F1 record" if name in F1_EVAL_ORIGINS else "time stamps", "expected_events": expected,
                    "passes": bool(expected >= N_MIN)}
    out["M3"] = m3
    # F1 stored forecasts: hashes and presence of arrays (values are not read)
    f1 = {}
    for name, want in F1_SHA.items():
        p = os.path.join(F1, name)
        f1[name] = {"exists": os.path.exists(p), "sha256_matches": os.path.exists(p) and sha(p) == want}
    for name in ("scores_validation.npz", "scores_final.npz"):
        z = np.load(os.path.join(F1, name), allow_pickle=False)
        f1[name]["pit_arrays"] = {f"{m}|h{h}": all(f"{m}|h{h}|{k}" in z.files for k in ("pit", "j")) for m, h in (("B7N", 1), ("B7N", 4), ("B4", 24))}
    out["f1_artifacts"] = f1
    out["seconds"] = time.time() - t0
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "a0_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("stamps_main", "time_basis_evaluation", "stamps_hist_supplementary", "origins_by_stamps", "labels_estimation", "M3",
                                          "f1_artifacts", "seconds")}, indent=None))


if __name__ == "__main__":
    main()
