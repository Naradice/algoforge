"""A1 stage S3 (plan section 7): data checks, series construction, leak test and the single scoring of the forward period. Primary comparison and the
alert-rate / Sunday sensitivities; the secondary analyses of plan section 8 are not in this script.

  python a1_s3.py            (refused before 2030-06-01, without all archives, or when results/evaluation.json already exists)
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

import a1_lib as L
import a1_data as D
fm, fc = L.fm, L.fc

HISTORY = ("2022-09", "2026-11")
FORWARD_FILES = ("2026-12", "2030-06")          # one extra file: the archives are split by Japan-time months
EARLIEST = date(2030, 6, 1)
OPEN_SHARE = 0.95
TIME_LIMIT = 3 * 3600
WORKERS = 12


def month_list(a, b):
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def required_months():
    return month_list(*HISTORY) + month_list(*FORWARD_FILES)


def extend_series(close, index, new_minutes):
    """main series unchanged, new 1-minute closes after its last row (plan section 3). new_minutes: list of Series, one per archive."""
    s = pd.concat(new_minutes)
    assert not s.index.duplicated().any(), "the same minute appears in two archives"
    assert s.index.is_monotonic_increasing, "minutes out of order across archives"
    s = s[s.index > index.max()]
    return np.concatenate([close, s.values.astype(float)]), index.append(pd.DatetimeIndex(s.index))


def weekly_open(index, a, b):
    """share of weeks (Monday in [a, b]) whose first row, Sunday-dated rows removed, is Monday 00:00-00:29."""
    x = index[index.dayofweek != 6]
    shifted = x + pd.Timedelta(hours=6)
    monday = shifted.normalize() - pd.to_timedelta(shifted.dayofweek.values, unit="D")
    m = (monday >= pd.Timestamp(a)) & (monday <= pd.Timestamp(b))
    first = pd.Series(x[m], index=monday[m]).groupby(level=0).min()
    ok = (first.dt.dayofweek.values == 0) & (first.dt.hour.values == 0) & (first.dt.minute.values < 30)
    return {"weeks": int(len(ok)), "conforming": int(ok.sum()), "share": float(ok.mean()) if len(ok) else 0.0}


_CTX = {}


def _context(path):
    if not _CTX:
        z = np.load(path)
        close, index = z["close"], pd.DatetimeIndex(z["index"].astype("datetime64[ns]"))
        edges = z["edges"]
        _CTX["v"] = (close, index, edges)
        _CTX["ctx"] = fm.Context(close, index, edges)
    return _CTX["ctx"]


def _month_job(args):
    path, ms, period = args
    fc.PERIODS["forward"] = period
    js, pit = L.month_pit(_context(path), "forward", ms)
    return js, pit


def _leak_job(args):
    path, picks, seed, period = args
    fc.PERIODS["forward"] = period
    ctx = _context(path)
    close, index, edges = _CTX["v"]
    F = L.Features(ctx)
    rng = np.random.default_rng(seed)
    states, bad = {}, []
    for j in picks:
        ms = pd.Timestamp(ctx.tt[j - 1].year, ctx.tt[j - 1].month, 1)
        if ms not in states:
            states[ms] = fm.build_state(ctx, ms, None)
        if not L.leak_one(close, index, edges, "forward", F, states[ms], int(j), rng):
            bad.append(int(j))
    return bad


def _state_job(args):
    path, ms, seed = args
    ctx = _context(path)
    close, index, edges = _CTX["v"]
    return str(ms.date()), L.state_unchanged(close, index, edges, ms, fm.build_state(ctx, ms, None), np.random.default_rng(seed))


def score_period(series_path, frozen, n_boot=L.BOOT, n_leak=L.LEAK_N, workers=WORKERS, deadline=None):
    """leak test, then ONE scoring of fc.PERIODS['forward']."""
    period = fc.PERIODS["forward"]
    ctx = _context(series_path)
    months = list(pd.date_range(pd.Timestamp(period[0]).replace(day=1), pd.Timestamp(period[1]), freq="MS"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_month_job, [(series_path, m, period) for m in months]))
        j = np.concatenate([r[0] for r in res])
        pit = dict(zip(j.tolist(), np.concatenate([r[1] for r in res]).tolist()))
        F = L.Features(ctx)
        inc = L.included_origins(F, "forward")
        picks = np.sort(np.random.default_rng(20261304).choice(inc, size=min(n_leak, len(inc)), replace=False))
        bad = sorted(sum(ex.map(_leak_job, [(series_path, part, 20261600 + i, period) for i, part in enumerate(np.array_split(picks, workers))]), []))
        changed = [m for m, ok in ex.map(_state_job, [(series_path, m, 20261700 + i) for i, m in enumerate(months)]) if not ok]
    leak = {"origins_tested": int(len(picks)), "mismatching_origins": bad, "months_state_changed": changed, "passed": bool(not bad and not changed)}
    if not leak["passed"]:
        return {"stopped": "leak test failed", "leak": leak}
    if deadline and time.time() > deadline:
        return {"stopped": "time limit before scoring", "leak": leak}
    tab = L.table(F, "forward", pit)
    out = {"leak": leak, "primary": L.compare(tab, frozen, L.ALPHA, n_boot)}
    out["sensitivity_alpha"] = {str(a): L.compare(tab, frozen, a, n_boot) for a in L.ALPHA_SENS}
    out["sensitivity_keep_sunday"] = L.compare(L.table(F, "forward", pit, drop_sunday=False), frozen, L.ALPHA, n_boot)
    for k in list(out["sensitivity_alpha"].values()) + [out["sensitivity_keep_sunday"]]:
        k.pop("judgement", None)                                                # only the primary comparison is judged
    np.savez(os.path.join(os.path.dirname(series_path), "evaluation_table.npz"), **{k: np.asarray(v) for k, v in tab.items() if k != "time"})
    return out


def load_frozen():
    frozen = json.load(open(os.path.join(L.RESULTS, "frozen.json"), encoding="utf-8"))
    assert L.sha256(frozen["reference_file"]) == frozen["reference_sha256"], "frozen reference values changed"
    frozen["_ref"] = dict(np.load(frozen["reference_file"]))
    return frozen


def main():
    from a1_s1 import code_state
    out_path = os.path.join(L.RESULTS, "evaluation.json")
    if date.today() < EARLIEST:
        raise SystemExit(f"S3 is not allowed before {EARLIEST}")
    if os.path.exists(out_path):
        raise SystemExit("the forward period was already scored; it is scored once")
    deadline = time.time() + TIME_LIMIT
    code, frozen = code_state(), load_frozen()
    ledger = json.load(open(os.path.join(L.RESULTS, "ledger.json"), encoding="utf-8"))
    arch = D.archives()
    need = required_months()
    missing = [m for m in need if m not in arch or m not in ledger]
    if missing:
        raise SystemExit(f"STOP: missing archives or ledger entries: {missing}")
    for m in need:
        assert L.sha256(arch[m]) == ledger[m]["sha256"], f"archive {m} differs from the ledger"
    chk = json.load(open(os.path.join(L.RESULTS, "check202208.json"), encoding="utf-8"))
    assert chk["identical"], "STOP: the 2022-08 check did not pass"
    close, index = L.load_main()
    with ProcessPoolExecutor(max_workers=6) as ex:                              # one archive per task; map keeps the month order
        minutes = list(ex.map(D.minute_closes, [arch[m] for m in need]))
    close, index = extend_series(close, index, minutes)
    opens = {"history": weekly_open(index, "2022-09-05", "2026-11-30"), "forward": weekly_open(index, *fc.PERIODS["forward"])}
    data = {"rows": int(len(close)), "backward_tick_steps": {m: ledger[m]["backward_steps"] for m in need if ledger[m]["backward_steps"]}, "weekly_open": opens}
    if min(o["share"] for o in opens.values()) < OPEN_SHARE:
        json.dump({"_code": code, "data": data, "stopped": "weekly open check"}, open(os.path.join(L.RESULTS, "evaluation_stopped.json"), "w"), indent=1)
        raise SystemExit(f"STOP: weekly open check failed: {opens}")
    os.makedirs(L.WORK, exist_ok=True)
    series_path = os.path.join(L.WORK, "series_forward.npz")
    np.savez(series_path, close=close, index=index.values.astype("datetime64[ns]").astype("int64"), edges=np.asarray(L.f1_edges()))
    res = score_period(series_path, frozen, deadline=deadline)
    res = {"_code": code, "data": data, "seconds": time.time() - (deadline - TIME_LIMIT), **res}
    name = "evaluation.json" if "primary" in res else "evaluation_stopped.json"
    json.dump(res, open(os.path.join(L.RESULTS, name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
