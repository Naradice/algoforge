"""A1 unit and end-to-end checks on synthetic inputs only (no research data is read).  python a1_test.py <scratch dir>"""
from __future__ import annotations

import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

import a1_lib as L
import a1_data as D
import a1_s3 as S3
fm, fc = L.fm, L.fc


def series(start, end, seed):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, end, freq="1min")
    t = t[(t.dayofweek < 5) | ((t.dayofweek == 6) & (t.hour >= 22) & (t >= "2022-01-01"))]
    keep = np.ones(len(t), bool)
    for i in rng.integers(0, len(t) - 40, len(t) // 4000):
        keep[i:i + rng.integers(1, 30)] = False
    t = t[keep]
    h = np.zeros(len(t) // 600 + 1)
    for i in range(1, len(h)):
        h[i] = 0.85 * h[i - 1] + 0.5 * rng.standard_normal()
    vol = 1e-4 * np.exp(0.5 * np.repeat(h, 600)[:len(t)])
    c = np.round(110 * np.exp(np.cumsum(vol * rng.standard_t(5, len(t)))), 3)
    return pd.Series(c, index=t, name="close")


def write_zip(path, minutes, seed):
    """tick archive whose last ASK per minute equals `minutes`; extra earlier ticks, a missing ASK and a BID column."""
    rng = np.random.default_rng(seed)
    lines = ["<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>"]
    for t, v in minutes.items():
        d = t.strftime("%Y.%m.%d")
        lines.append(f"{d}\t{t.strftime('%H:%M')}:01.000\t{v - 0.5:.3f}\t{v + rng.integers(1, 9) / 1000:.3f}\t\t")
        lines.append(f"{d}\t{t.strftime('%H:%M')}:30.000\t{v - 0.4:.3f}\t{v:.3f}\t\t")
        lines.append(f"{d}\t{t.strftime('%H:%M')}:45.000\t{v - 0.3:.3f}\t\t\t")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(os.path.basename(path).replace(".zip", ".csv"), "\n".join(lines) + "\n")


def unit(tmp):
    rng = np.random.default_rng(1)
    s = series("2021-01-04", "2021-06-30", 2)
    s.iloc[:0]
    ctx = fm.Context(s.values, s.index, np.linspace(0.001, 0.05, 19))
    F = L.Features(ctx)
    fc.PERIODS["forward"] = ("2021-03-01", "2021-06-15")
    t = ctx.s.t
    lc = np.log(s.values)
    inc = L.included_origins(F, "forward")
    # brute force of the origin rule, label, features
    sec = s.index.values.astype("datetime64[s]").astype("int64")
    gapfree = lambda a, b: a >= 0 and b < len(s) and np.all(np.diff(sec[a:b + 1]) <= 300)
    lo, hi = pd.Timestamp("2021-03-01"), pd.Timestamp("2021-06-16")

    def valid(j):
        r = t[j]
        if j < 1 or not (lo <= s.index[r] < hi) or not gapfree(r - 59, r) or not gapfree(t[j - 1], r) or not gapfree(r, r + 60) or not s.index[r + 60] < hi:
            return False
        return np.sum(np.diff(lc[t[j - 1]:r + 1]) ** 2) > 0
    want = [j for j in range(1, ctx.s.nblocks) if valid(j) and valid(j - 1) and not np.any(s.index[max(t[j] - 119, 0):t[j] + 61].dayofweek == 6)]
    assert list(inc) == want, (len(inc), len(want))
    blocks = [j for j in range(ctx.s.nblocks) if gapfree(t[j] - 59, t[j])]
    rv = {j: np.sum(np.diff(lc[t[j] - 59:t[j] + 1]) ** 2) for j in blocks}
    for j in rng.choice(inc, 40, replace=False):
        got, c = F.scores(j)
        past = [b for b in blocks if b <= j]
        assert np.isclose(got[0], rv[j], rtol=1e-9)
        for i, k in ((1, 4), (2, 24), (3, 120)):
            assert np.isclose(got[i], np.mean([rv[b] for b in past[-k:]]), rtol=1e-9)
        assert np.isclose(got[4], np.mean(np.abs(np.diff(s.values[t[j] - 59:t[j] + 1])))) and np.isclose(got[5], abs(lc[t[j]] - lc[t[j - 1]]))
        ms = pd.Timestamp(s.index[t[j]].year, s.index[t[j]].month, 1)
        old = [b for b in blocks if s.index[t[b]] < ms]
        hour = (s.index[t[j]] + pd.Timedelta(hours=1) - pd.Timedelta(hours=7)).hour
        same = [rv[b] for b in old if (s.index[t[b]] - pd.Timedelta(hours=7)).hour == hour]
        fac = np.mean(same) / np.mean([rv[b] for b in old]) if same else 1.0
        assert np.isclose(got[6], got[2] * fac, rtol=1e-9)
        prior = [b for b in range(1, j) if np.isfinite(ctx.r[b])][-24:]
        assert np.isclose(c, abs(ctx.r[j]) / np.std([lc[t[b]] - lc[t[b] - 60] for b in prior]), rtol=1e-9)
        assert np.isclose(F.label_value(j), np.sum(np.diff(lc[t[j]:t[j] + 61]) ** 2), rtol=1e-9)
    # events and alerts
    row = np.array([0, 60, 120, 420, 480, 900])
    y = np.array([True, False, True, True, True, True])
    assert list(L.event_onsets(row, y)) == [0, 3, 5], L.event_onsets(row, y)
    for _ in range(200):
        n = 60
        sc = rng.integers(0, 8, n).astype(float)
        on = rng.random(n) < 0.15
        w = rng.integers(0, 4, n).astype(float)
        order = L.alert_order(sc)
        expanded = [i for i in sorted(range(n), key=lambda i: (-sc[i], i)) for _ in range(int(w[i]))]
        k = int(np.floor(0.1 * len(expanded) + 1e-9))
        total = sum(w[on])
        got = L.detection_rate(order, on, 0.1, w)
        assert (np.isnan(got) and total == 0) or np.isclose(got, sum(on[i] for i in expanded[:k]) / total)
    assert L.judge([0.06, 0.2], [0.01, 0.2], 150) == "支持" and L.judge([0.06, 0.2], [-0.01, 0.2], 150) == "判定保留"
    assert L.judge([-0.04, 0.04], [-1, 1], 150) == "上乗せなし" and L.judge([0.2, 0.3], [0.1, 0.2], 99) == "判定保留"
    # tick archive -> minutes, ledger from time stamps only
    m = s["2021-03-01":"2021-03-03"]
    p = os.path.join(tmp, "ticks_USDJPY-oj5k_2021-03.zip")
    write_zip(p, m, 3)
    got = D.minute_closes(p)
    assert (got.index == m.index).all() and np.allclose(got.values, m.values, atol=1e-9)
    e = D.ledger_entry(p)
    assert e["tick_rows"] == 3 * len(m) and e["minutes"] == len(m) and e["backward_steps"] == 0
    assert e["sunday_tick_rows"] == 3 * int((m.index.dayofweek == 6).sum())
    o = S3.weekly_open(s.index, "2021-02-01", "2021-05-31")
    assert o["weeks"] > 10 and o["share"] > 0.8, o
    assert len(S3.required_months()) == 51 + 19
    print("unit checks passed", flush=True)


def end_to_end(tmp):
    """synthetic series split into a 'main' part and monthly archives -> extended series -> leak test and one scoring."""
    s = series("2021-09-01", "2022-12-30", 7)
    cut = pd.Timestamp("2022-08-31 17:59")
    old, new = s[s.index <= cut], s[(s.index > cut) & (s.index < "2022-11-01")]
    parts = []
    for k, (month, g) in enumerate(new.groupby(new.index.strftime("%Y-%m"))):
        p = os.path.join(tmp, f"ticks_USDJPY-oj5k_{month}.zip")
        write_zip(p, g, 10 + k)
        parts.append(D.minute_closes(p))
    close, index = S3.extend_series(old.values, old.index, parts)
    full = s[s.index < "2022-11-01"]
    assert len(close) == len(full) and (index == full.index).all() and np.allclose(close, full.values, atol=1e-9)
    path = os.path.join(tmp, "series_forward.npz")
    np.savez(path, close=close, index=index.values.astype("datetime64[ns]").astype("int64"), edges=np.linspace(0.001, 0.05, 19))
    fc.PERIODS["forward"] = ("2022-09-01", "2022-10-31")
    rng = np.random.default_rng(4)
    ref = {k: np.sort(rng.random(500) * (1e-5 if k == "base" else 1)) for k in ("base", "S", "C")}
    frozen = {"baseline": "rv24", "coef_s": [-3.0, 2.0, 0.5], "coef_c": [-3.0, 2.0, 0.5], "_ref": ref}
    out = S3.score_period(path, frozen, n_boot=100, n_leak=12, workers=3)
    assert out["leak"]["passed"], out["leak"]
    pr = out["primary"]
    assert pr["origins"] > 300 and pr["events"] > 0 and len(pr["ci_delta"]) == 2 and "judgement" in pr, pr
    assert pr["judgement"] == "判定保留" or pr["events"] >= L.N_MIN
    print("end-to-end check passed:", json.dumps({k: pr[k] for k in ("origins", "events", "alerts")}), flush=True)


if __name__ == "__main__":
    tmp = sys.argv[1]
    os.makedirs(tmp, exist_ok=True)
    unit(tmp)
    end_to_end(tmp)
