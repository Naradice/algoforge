"""Stage A0 (plan v0.6 section 3): reinterpretation of the existing atlas and of the P5-B real-window statistics. No new simulation, no new real-data analysis.
Output: results/a0.json"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dr_env as E  # noqa: E402

STATS = ("std", "skewness", "kurtosis", "acf_1", "acf_5", "acf_20", "volclust_50")
MAP = {"acf_1": "acf_1", "kurtosis": "kurtosis", "volclust_50": "volclust_50"}


def main():
    code = E.code_state()
    a = pd.read_csv(E.ATLAS)
    v = a.dropna(subset=["std", "acf_1", "volclust_50"]).copy()
    out = {"_code": code, "n_atlas": int(len(a)), "n_valid": int(len(v)), "n_stable_all_5_seeds": int((a["success_rate"] == 1.0).sum()), "stats": {}}
    for k in STATS:
        x = v[k].astype(float)
        sd_pts = float(x.std(ddof=1))
        se = 0.5 * v[f"{k}_std_across_seeds"].astype(float)          # np.nanstd(ddof=0) over 5 seeds -> SE = sd_unbiased / sqrt(5) = 0.5 * sd0
        out["stats"][k] = {"min": float(x.min()), "p25": float(x.quantile(.25)), "median": float(x.median()), "p75": float(x.quantile(.75)), "max": float(x.max()),
                           "sd_across_points": sd_pts, "median_se_seed": float(se.median()), "median_se_over_sd_across_points": float((se / sd_pts).median()),
                           "frac_points_se_le_0.25_sd_across_points": float(np.mean(se <= 0.25 * sd_pts))}
    corr = v[list(STATS)].corr(method="spearman")
    out["spearman_across_points"] = {r: {c: float(corr.loc[r, c]) for c in STATS} for r in STATS}
    bands = json.load(open(E.BANDS, encoding="utf-8"))["buckets"]
    rows = []
    for q, b in bands.items():
        m, sd = b["mean"], b["window_sd"]
        inbox = {}
        for k in ("acf_1", "kurtosis", "volclust_50"):
            inbox[k] = (np.abs(v[k] - m[k]) <= 0.5 * sd[k]).values
        rows.append({"q": int(q), **{f"real_{k}": float(m[k]) for k in ("acf_1", "kurtosis", "volclust_50")},
                     **{f"atlas_range_contains_{k}": bool(v[k].min() <= m[k] <= v[k].max()) for k in ("acf_1", "kurtosis", "volclust_50")},
                     **{f"n_atlas_in_half_sd_box_{k}": int(inbox[k].sum()) for k in inbox}, "n_atlas_in_3d_box": int((inbox["acf_1"] & inbox["kurtosis"] & inbox["volclust_50"]).sum()),
                     "n_atlas_in_2d_box_acf1_kurt": int((inbox["acf_1"] & inbox["kurtosis"]).sum())})
    out["real_buckets_vs_atlas"] = rows
    out["note"] = "bands.json window SDs come from overlapping windows (P5-B): reference only, not the independent windows of this study; the box here is 0.5 x that SD."
    json.dump(out, open(os.path.join(E.RESULTS, "a0.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("A0 written")


if __name__ == "__main__":
    main()
