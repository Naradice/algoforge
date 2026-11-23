"""P5-B stage 3 step 1 (plan v0.3 gate 4): real-side window statistics for all 20 buckets and the bands, fixed BEFORE the analysis. Output: results/bands.json.
Bands are descriptive (window-to-window variation of overlapping windows; not confidence intervals, no coverage meaning). Run, then commit results/bands.json; s3_analysis refuses to run on a dirty tree."""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s3_env as E  # noqa: E402
import s3_lib as L  # noqa: E402
import p4_lib as P  # noqa: E402

KEYS = ("std", "kurtosis", "acf_1", "acf_5", "acf_20", "volclust_50")


def wstats(closes):
    s = L.S2.path_stats(closes)
    return [s["atlas"][k] for k in KEYS] + [s["vr_win"]["60"]]


def main():
    code = E.code_state()
    mp = E.load_mapping()
    raw = P.V.Data().raw
    tasks = []
    for b in mp["buckets"]:
        for blk in b["used_blocks"]:
            tasks.append(raw.close[blk * 60:blk * 60 + 1001])
    with ProcessPoolExecutor(max_workers=E.WORKERS) as ex:
        res = np.array(list(ex.map(wstats, tasks, chunksize=20)))
    out, pos, worst = {}, 0, 0.0
    for b in mp["buckets"]:
        n = len(b["used_blocks"])
        w = res[pos:pos + n]
        pos += n
        d = {k: w[:, i] for i, k in enumerate(KEYS)}
        vrw = w[:, len(KEYS)]
        mean = {k: float(d[k].mean()) for k in KEYS}
        sd = {k: float(d[k].std(ddof=1)) for k in KEYS}
        for k in ("std", "acf_1", "acf_5", "acf_20", "volclust_50", "kurtosis"):
            worst = max(worst, abs(mean[k] - b["real_stats"][k]) / max(abs(b["real_stats"][k]), 1e-12))
        per_win_proxy = L.vr_proxy(d["acf_1"], d["acf_5"], d["acf_20"])
        center = float(L.vr_proxy(mean["acf_1"], mean["acf_5"], mean["acf_20"]))
        vr_sd = float(per_win_proxy.std(ddof=1))
        out[str(b["q"])] = {
            "n_windows": int(n), "mean": mean, "window_sd": sd, "vr_win60_mean": float(vrw.mean()), "vrproxy_from_mean_acf": center, "vrproxy_window_sd": vr_sd,
            "bands": {"std_sd": [mean["std"] - sd["std"], mean["std"] + sd["std"]], "std_pct25": [0.75 * mean["std"], 1.25 * mean["std"]],
                      "acf_1_sd": [mean["acf_1"] - sd["acf_1"], mean["acf_1"] + sd["acf_1"]], "vrproxy_sd": [center - vr_sd, center + vr_sd]}}
    assert worst < 1e-9, worst
    json.dump({"_code": code, "note": "real-side windows: 300 used windows per bucket (overlapping); SD ddof=1 across windows; descriptive bands, not confidence intervals",
               "gate_window_means_equal_mapping_real_stats_max_rel_diff": worst, "buckets": out}, open(E.BAND_FILE, "w", encoding="utf-8"), indent=1)
    print("bands written; max rel diff to mapping real_stats", worst, flush=True)


if __name__ == "__main__":
    main()
