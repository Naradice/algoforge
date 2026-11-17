"""T0-D: read the event times of Fig.3 (left panels) from the vector drawing of arXiv physics/0608099.

Fig.3 left draws each transaction as a short vertical tick on a 0-500 s axis (x pixel 1024.52 -> 0 s,
2865.74 -> 500 s on page 6, units of 0.1 pt). The upper (grey) tick row is model-1, the middle (black)
row is model-2. Prints interval statistics of the two excerpts.

Usage: python research/tdm_fig3_pdf_extract.py <paper.pdf> [out_prefix]
"""
import re
import sys

import numpy as np
import pypdf

X0, X500 = 1024.52, 2865.74     # axis ticks read from the same page ("0" origin = left axis line, "500" tick)


def tick_times(pdf_path):
    page = pypdf.PdfReader(pdf_path).pages[5]
    c = re.sub(r"BT.*?ET", "", page.get_contents().get_data().decode("latin1"), flags=re.S)
    nums, cur, paths, lw = [], [], [], None
    for t in c.split():
        if re.fullmatch(r"-?\d*\.?\d+", t):
            nums.append(float(t)); continue
        if t == "m":
            cur = [(nums[-2], nums[-1])]
        elif t == "l":
            cur.append((nums[-2], nums[-1]))
        elif t == "w":
            lw = nums[-1]
        elif t in ("S", "s") and len(cur) == 2:
            paths.append((lw, cur)); cur = []
        nums = []
    ticks = [p for lw, p in paths if abs(lw - 8.63071) < 1e-3 and abs(p[0][0] - p[1][0]) < 0.5]
    t_up = np.sort([(p[0][0] - X0) / (X500 - X0) * 500 for p in ticks if p[0][1] > 6800])
    t_mid = np.sort([(p[0][0] - X0) / (X500 - X0) * 500 for p in ticks if p[0][1] < 6600])
    return t_up, t_mid


def describe(name, t):
    d = np.diff(t)
    print(f"{name}: n={len(t)} span {t.min():.1f}-{t.max():.1f} s  mean {d.mean():.2f}  median {np.median(d):.2f}  "
          f"CV {d.std() / d.mean():.2f}  max/mean {d.max() / d.mean():.1f}  ACF1 {np.corrcoef(d[:-1], d[1:])[0, 1]:.2f}")


if __name__ == "__main__":
    up, mid = tick_times(sys.argv[1])
    describe("model-1 (grey row)", up)
    describe("model-2 (black row)", mid)
    print("events per tau=150 s window in model-2:", round(150 / np.diff(mid).mean(), 1))
    if len(sys.argv) > 2:
        np.save(sys.argv[2] + "_model1.npy", up)
        np.save(sys.argv[2] + "_model2.npy", mid)
