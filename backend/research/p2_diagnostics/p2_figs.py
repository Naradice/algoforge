"""P2 figures from p2_results.json (PIT histograms with week-block 95% bands; coverage curves; k90 by ATR group)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = r"C:\tmp\p2_out"
d = json.load(open(os.path.join(OUT, "p2_results.json"), encoding="utf-8"))
cells = list(d["cells"].keys())
fig, ax = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
for a, k in zip(ax.ravel(), cells):
    h = d["cells"][k]["D1"]["hist_expected_mass_with_weekblock_ci"]
    x = (np.arange(20) + 0.5) / 20
    est = np.array([b["est"] for b in h]); lo = np.array([b["ci"][0] for b in h]); hi = np.array([b["ci"][1] for b in h])
    a.bar(x, est, width=0.045, color="#4c72b0"); a.errorbar(x, est, yerr=[est - lo, hi - est], fmt="none", ecolor="k", lw=0.8)
    a.axhline(0.05, color="r", ls="--"); a.set_title(k)
fig.suptitle("PIT histogram (expected mass over tie randomisation; bars = week-block 95% CI), uniform = 0.05")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pit_hist.png"), dpi=110)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, per in zip(ax, ("validation", "final")):
    for h in (1, 4, 12, 24):
        c = d["cells"][f"{per}|h{h}"]["D2"]["coverage"]
        a.plot([50, 80, 90, 95], [c[x]["est"] for x in ("50", "80", "90", "95")], marker="o", label=f"h={h}")
    a.plot([50, 95], [0.5, 0.95], "k--", lw=0.8); a.set_title(per); a.set_xlabel("nominal %"); a.set_ylabel("coverage"); a.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_coverage.png"), dpi=110)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, per in zip(ax, ("validation", "final")):
    for h in (1, 4, 12, 24):
        g = d["cells"][f"{per}|h{h}"]["D4"]["atr5"]["descriptive"]
        a.plot(range(5), [g[str(i)]["k90"]["k"] for i in range(5)], marker="o", label=f"h={h}")
    a.axhline(1, color="k", ls="--", lw=0.8); a.set_title(per + " k90 by ATR group (0=low)"); a.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_k90_atr.png"), dpi=110)
print("ok")
