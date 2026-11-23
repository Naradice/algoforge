"""P5-B stage 3 figures (plan v0.3 3.2 and 3.4): scatter plots from stored results only. Output: results/fig_a3_std_acf1.png, fig_c3_C3.png, fig_c3_C0s.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s3_env as E  # noqa: E402
import s3_lib as L  # noqa: E402

d = json.load(open(os.path.join(E.RESULTS, "analysis3.json"), encoding="utf-8"))
bands = json.load(open(E.BAND_FILE, encoding="utf-8"))["buckets"]
atlas = pd.read_csv(E.ATLAS)
valid = L.valid_points(atlas)
fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
for ax, q in zip(axs.ravel(), E.BUCKETS4):
    b, bb = bands[str(q)], bands[str(q)]["bands"]
    ax.scatter(valid["std"], valid["acf_1"], s=10, c="lightgray", label="atlas valid points (170)")
    sel, cf = d["B3"][str(q)]["selected"], d["B3"][str(q)]["CF1"]["selected"]
    ax.scatter([s["std_ratio"] * b["mean"]["std"] for s in sel], [s["acf_1"] for s in sel], s=28, c="tab:blue", label="current selection (8)")
    ax.scatter([s["std_ratio"] * b["mean"]["std"] for s in cf], [s["acf_1"] for s in cf], s=28, marker="^", c="tab:orange", label="CF1 (8, diagnostic)")
    ax.axvspan(*bb["std_sd"], color="tab:green", alpha=0.25, label="std band (+-1 window SD)")
    ax.axvspan(*bb["std_pct25"], color="tab:green", alpha=0.10, label="std +-25%")
    ax.axhspan(*bb["acf_1_sd"], color="tab:red", alpha=0.12, label="acf_1 band (+-1 window SD)")
    ax.plot([b["mean"]["std"]], [b["mean"]["acf_1"]], "k*", ms=11, label="real (mean of windows)")
    ax.set_xscale("log")
    ax.set_title(f"bucket {q}")
    ax.set_xlabel("std (log scale)")
    ax.set_ylabel("acf_1")
axs[0, 0].legend(fontsize=7)
fig.suptitle("A3: atlas points vs real bands (bands = window-to-window variation of overlapping windows, not confidence intervals)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(E.RESULTS, "fig_a3_std_acf1.png"), dpi=110)
plt.close(fig)
for cond in E.CONDS:
    rows = pd.DataFrame(d["C3"]["conds"][cond]["rows"])
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    cols = {0: "tab:blue", 8: "tab:green", 15: "tab:orange", 19: "tab:red"}
    for ax, (x, lab) in zip(axs, (("atlas_kurtosis", "atlas kurtosis (log)"), ("rel_seed_sd", "atlas std_std / std"), ("atlas_acf_1", "atlas acf_1"))):
        for q in E.BUCKETS4:
            r = rows[rows["q"] == q]
            ax.scatter(r[x], r["log_V_ens_over_sigma2_atlas"], c=cols[q], s=22, label=f"bucket {q}")
        if x == "atlas_kurtosis":
            ax.set_xscale("log")
        ax.axhline(0, c="k", lw=0.6)
        ax.set_xlabel(lab)
        ax.set_ylabel("log(V_ens / sigma2_atlas)")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"C3 ({cond}): 32 observations (4 buckets x 8 candidates; duplicates of (delta, wma) exist); descriptive only", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(E.RESULTS, f"fig_c3_{cond}.png"), dpi=110)
    plt.close(fig)
print("figures written")
