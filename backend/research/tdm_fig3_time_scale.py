"""T0-F (Study Manager #886): seconds conversion of the paper's dt and c_i.

Read from the PDF drawing of arXiv physics/0608099 (pixel read-out, no data table in the paper):
  Fig.2  (page 5)  real USD/JPY transaction ticks on a 0-500 s axis      -> real mean interval
  Fig.3  (page 6)  model-1 / model-2 tick rows on a 0-500 s axis           -> see tdm_fig3_pdf_extract.py
  Fig.5F (page 8)  cumulative (complementary) distribution of the transaction interval [s], real vs model-4
and compares them with the literal reading of the captions (dt seconds per step, c_i per second) using the
paper's Model-1 (tdm_fig3_audit.model1_intervals).

Usage: python research/tdm_fig3_time_scale.py <paper.pdf> [--sweep]
"""
import re
import sys

import numpy as np
import pypdf


def _paths(page, ops=("S", "s")):
    c = re.sub(r"BT.*?ET", "", page.get_contents().get_data().decode("latin1"), flags=re.S)
    nums, cur, out, lw = [], [], [], None
    for t in c.split():
        if re.fullmatch(r"-?\d*\.?\d+", t):
            nums.append(float(t)); continue
        if t == "m":
            cur = [(nums[-2], nums[-1])]
        elif t == "l":
            cur.append((nums[-2], nums[-1]))
        elif t == "w":
            lw = nums[-1]
        elif t in ops and cur:
            out.append((lw, cur)); cur = []
        nums = []
    return out


def fig2_real(pdf):
    """62 vertical ticks at y~3834 on page 5; axis: label x + 43 = tick x, 345.2 px per 100 s, 0 s at 2291.8."""
    paths = _paths(pdf.pages[4])
    ev = [p for lw, p in paths if len(p) == 2 and abs(p[0][0] - p[1][0]) < 0.5 and round(min(p[0][1], p[1][1])) == 3834]
    t = np.sort([(p[0][0] - 2291.8) / 345.2 * 100 for p in ev])
    return t


def fig5f_curves(pdf):
    """The two 11-point polylines of panel F: x 20 s = 391.4 px from tick 0 at 3391.81; y: 1 at 4727.25, 186.0 px/decade."""
    paths = _paths(pdf.pages[7])
    out = []
    for lw, p in paths:
        if len(p) == 11 and all(3300 <= x <= 5300 and 3700 <= y <= 4800 for x, y in p):
            a = np.array(p)
            t = (a[:, 0] - 3391.81) / 391.4 * 20
            logp = (a[:, 1] - 4727.25) / 186.0
            out.append((t[::-1], 10 ** logp[::-1]))
    return out


def mean_from_ccdf(t, p):
    t = np.asarray(t); p = np.asarray(p)
    m = 0.0
    for i in range(len(t) - 1):
        dt, p1, p2 = t[i + 1] - t[i], p[i], p[i + 1]
        m += dt * (p1 - p2) / np.log(p1 / p2) if p1 != p2 else dt * p1
    return m


def main():
    pdf = pypdf.PdfReader(sys.argv[1])
    t = fig2_real(pdf)
    d = np.diff(t)
    print(f"Fig.2 real data: n={len(t)} ticks {t.min():.1f}-{t.max():.1f} s  mean interval {d.mean():.2f} s  CV {d.std() / d.mean():.2f}  "
          f"ACF1 {np.corrcoef(d[:-1], d[1:])[0, 1]:.2f}")
    for k, (tt, pp) in enumerate(fig5f_curves(pdf)):
        print(f"Fig.5F curve {k}: P(T>10 s)={np.interp(10, tt, pp):.3f}  P(T>30 s)={np.interp(30, tt, pp):.4f}  "
              f"mean interval (area under the complementary CDF, last point 80+ s truncated) ~ {mean_from_ccdf(tt, pp):.1f} s")
    if "--sweep" in sys.argv:
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tdm_fig3_audit import model1_intervals
        print("literal reading (dt = seconds per step, c_i per second): Model-1 interval statistics")
        for N, c, dt in ((100, 0.01, 0.01), (100, 0.015, 0.01), (150, 0.01, 0.01), (200, 0.01, 0.01), (500, 0.01, 0.01),
                         (100, 0.02, 1.0)):
            m, cv, a1 = [], [], []
            for seed in (1, 2, 3):
                _, Tp, _ = model1_intervals(6000, N, 1.0, dt, c, "signed", seed)
                m.append(Tp.mean()); cv.append(Tp.std() / Tp.mean()); a1.append(np.corrcoef(Tp[:-1], Tp[1:])[0, 1])
            print(f"  N={N:4d} c={c:5.3f} dt={dt:<5g} signed: mean {np.mean(m):6.2f} s  CV {np.mean(cv):.2f}  ACF1 {np.mean(a1):+.2f}")


if __name__ == "__main__":
    main()
