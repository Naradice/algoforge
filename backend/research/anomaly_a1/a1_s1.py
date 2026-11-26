"""A1 stage S1 (plan section 7): durable copy of the F1 run, bit-for-bit reproduction of the stored validation PIT, estimation-period forecasts,
leak test, fit of the combinations on estimation and choice of the baseline on validation. Only the estimation and validation periods are used;
no file of the evaluation period is read.

  python a1_s1.py copy | reproduce | leak | freeze        (run in this order; each stage requires the previous result)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import a1_lib as L
fm, fc = L.fm, L.fc

F1_SOURCE = r"C:\tmp\f1_run1_db0926f"
F1_SHA = {"selection.json": "735d28f4c2865a99069a3d6e2b1613cf86f1cc3f51e526937945a68ea988d13a",
          "scores_final.npz": "e08b30c6fc6b1b69ab9a26f670b497d62381fe06a623ea2caad639f90bbfcc15",
          "scores_validation.npz": "65a3c46a70e7bdf837c29af1454022e5750016a4e02e9c4ac595187d50451046",
          "ddm_mapping.json": "962a9843353d81e626ad4188018d778056ec02e4d048fae37b6069341a01201a",
          "ddm_ens_seed1.npy": "074d973a16f6884b5e327ee6e14f5a1df7f465914dbb47d5791c04761132605a",
          "ddm_ens_seed2.npy": "7c07a073a4ab2261cf4109bcebdca9f5acf9a61b7e5aca6bf6cc36bca7a0efa2",
          "ddm_ens_seed3.npy": "73b14d992523ab0ae3374ee4b10be3dcce3103494751776d0d90ee64e2c2cdbb"}
VERSIONS = {"arch": "8.0.0", "numpy": "2.3.3", "scipy": "1.17.1", "pandas": "3.0.1", "statsmodels": "0.14.6"}
WORKERS = 12
SINCE = {"estimation": L.FIT_START, "validation": None}


def code_state():
    paths = ["backend/research/anomaly_a1", "backend/research/f1_common.py", "backend/research/f1_models.py", "docs/research-anomaly-a1-plan.md"]
    git = lambda *a: subprocess.check_output(["git", "-C", L.REPO, *a], text=True).strip()
    dirty = git("status", "--porcelain", "--", *paths, ":(exclude)backend/research/anomaly_a1/results")
    if dirty:
        raise SystemExit("uncommitted changes -> commit first:\n" + dirty)
    diff = git("diff", "--stat", "db0926f1afd6353339a90abeeac9717a400afe4b", "HEAD", "--", "backend/research/f1_common.py", "backend/research/f1_models.py")
    assert not diff, "F1 code changed since db0926f: " + diff
    import importlib.metadata as md
    got = {k: md.version(k) for k in VERSIONS}
    assert got == VERSIONS, got
    return {"head": git("rev-parse", "HEAD"), "versions": got, "python": sys.version.split()[0], "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def save(name, obj):
    os.makedirs(L.RESULTS, exist_ok=True)
    json.dump(obj, open(os.path.join(L.RESULTS, name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def load(name):
    return json.load(open(os.path.join(L.RESULTS, name), encoding="utf-8"))


_CTX = {}


def context():
    if not _CTX:
        close, index = L.load_main()
        _CTX["v"] = (close, index, L.f1_edges())
        _CTX["ctx"] = fm.Context(close, index, _CTX["v"][2])
    return _CTX["ctx"]


def month_job(args):
    period, ms = args
    js, pit = L.month_pit(context(), period, ms)
    return period, str(ms.date()), js, pit


def months(period):
    start = L.FIT_START if period == "estimation" else fc.PERIODS[period][0]
    # estimation starts one month earlier so that the first fit month has the forecast of its previous origin
    if period == "estimation":
        start = str((pd.Timestamp(start) - pd.offsets.MonthBegin(1)).date())
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(fc.PERIODS[period][1]), freq="MS"))


def forecasts(period):
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(month_job, [(period, m) for m in months(period)]))
    j = np.concatenate([r[2] for r in res])
    p = np.concatenate([r[3] for r in res])
    order = np.argsort(j)
    return j[order], p[order]


def pit_path(period):
    return os.path.join(L.WORK, f"pit_{period}.npz")


def stage_copy(code):
    os.makedirs(L.F1_DIR, exist_ok=True)
    out = {}
    for name, want in F1_SHA.items():
        dst = os.path.join(L.F1_DIR, name)
        if not os.path.exists(dst):
            assert L.sha256(os.path.join(F1_SOURCE, name)) == want, name
            shutil.copy2(os.path.join(F1_SOURCE, name), dst)
        out[name] = L.sha256(dst) == want
    n = 0
    for name in sorted(os.listdir(F1_SOURCE)):                                  # the remaining run records
        src, dst = os.path.join(F1_SOURCE, name), os.path.join(L.F1_DIR, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
        n += os.path.isfile(dst)
    save("s1_copy.json", {"_code": code, "destination": L.F1_DIR, "sha256_matches": out, "files": n})
    assert all(out.values()), out
    print("copied", n, "files; hashes match")


def stage_reproduce(code):
    assert all(load("s1_copy.json")["sha256_matches"].values())
    os.makedirs(L.WORK, exist_ok=True)
    t0 = time.time()
    j, p = forecasts("validation")
    z = np.load(os.path.join(L.F1_DIR, "scores_validation.npz"))
    sj, sp = z["B7N|h1|j"], z["B7N|h1|pit"]
    same_set = bool(np.array_equal(j, sj))
    n_diff = int(np.sum(p != sp)) if same_set else None
    res = {"_code": code, "n_origins": int(len(j)), "n_stored": int(len(sj)), "same_origin_set": same_set, "n_pit_differences": n_diff,
           "bit_identical": bool(same_set and n_diff == 0), "seconds_validation": time.time() - t0}
    save("s1_reproduce.json", res)
    print(res)
    if not res["bit_identical"]:
        raise SystemExit("STOP: the stored validation PIT is not reproduced")
    np.savez(pit_path("validation"), j=j, pit=p)
    t0 = time.time()
    j, p = forecasts("estimation")
    np.savez(pit_path("estimation"), j=j, pit=p)
    ctx = context()
    F = L.Features(ctx)
    missing = {}
    for period in ("estimation", "validation"):
        inc = L.included_origins(F, period, since=SINCE[period])
        have = set(np.load(pit_path(period))["j"].tolist())
        missing[period] = {"included_origins": int(len(inc)), "without_previous_forecast": int(sum((x - 1) not in have for x in inc))}
    res.update({"estimation_forecasts": int(len(j)), "seconds_estimation": time.time() - t0, "missing": missing,
                "sha256": {p_: L.sha256(pit_path(p_)) for p_ in ("estimation", "validation")}})
    save("s1_reproduce.json", res)
    print({k: res[k] for k in ("estimation_forecasts", "missing")})


def leak_job(args):
    period, picks, seed = args
    ctx = context()
    close, index, edges = _CTX["v"]
    F = L.Features(ctx)
    rng = np.random.default_rng(seed)
    states, bad = {}, []
    for j in picks:
        ms = pd.Timestamp(ctx.tt[j - 1].year, ctx.tt[j - 1].month, 1)
        if ms not in states:
            states[ms] = fm.build_state(ctx, ms, None)
        if not L.leak_one(close, index, edges, period, F, states[ms], int(j), rng):
            bad.append(int(j))
    return period, bad


def state_job(args):
    ms, seed = args
    ctx = context()
    close, index, edges = _CTX["v"]
    st = fm.build_state(ctx, ms, None)
    return str(ms.date()), L.state_unchanged(close, index, edges, ms, st, np.random.default_rng(seed))


def stage_leak(code):
    assert load("s1_reproduce.json")["bit_identical"]
    ctx = context()
    F = L.Features(ctx)
    jobs, counts = [], {}
    for k, period in enumerate(("estimation", "validation")):
        inc = L.included_origins(F, period, since=SINCE[period])
        picks = np.sort(np.random.default_rng(20261302 + k).choice(inc, size=L.LEAK_N, replace=False))
        counts[period] = int(len(picks))
        jobs += [(period, part, 20261400 + 100 * k + i) for i, part in enumerate(np.array_split(picks, WORKERS))]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(leak_job, jobs))
        ms = months("estimation") + months("validation")
        st = list(ex.map(state_job, [(m, 20261500 + i) for i, m in enumerate(ms)]))
    bad = {p: sorted(b for per, bs in res if per == p for b in bs) for p in counts}
    out = {"_code": code, "origins_tested": counts, "mismatching_origins": bad, "months_state_tested": len(st),
           "months_state_changed": [m for m, ok in st if not ok]}
    out["passed"] = bool(not any(bad.values()) and not out["months_state_changed"])
    save("s1_leak.json", out)
    print(out)
    if not out["passed"]:
        raise SystemExit("STOP: leak test failed")


def stage_freeze(code):
    assert load("s1_leak.json")["passed"]
    ctx = context()
    F = L.Features(ctx)
    tabs = {}
    for period in ("estimation", "validation"):
        z = np.load(pit_path(period))
        pit = dict(zip(z["j"].tolist(), z["pit"].tolist()))
        tabs[period] = L.table(F, period, pit, since=SINCE[period])
    # baseline: largest detection rate of the candidate ALONE on validation at alpha = 0.05 (ties: smaller number)
    v = tabs["validation"]
    onsets = L.event_onsets(v["row"], v["y"])
    mask = np.zeros(len(v["row"]), bool)
    mask[onsets] = True
    rates = [L.detection_rate(L.alert_order(v["B"][:, i]), mask, L.ALPHA) for i in range(len(L.BASELINES))]
    best = int(np.argmax(rates))
    e = tabs["estimation"]
    ref = {"base": np.sort(e["B"][:, best]), "S": np.sort(e["S"]), "C": np.sort(e["C"])}
    ub, us, uc = L.ecdf(ref["base"], e["B"][:, best]), L.ecdf(ref["S"], e["S"]), L.ecdf(ref["C"], e["C"])
    coef_s, coef_c = L.fit_logit(e["y"], ub, us), L.fit_logit(e["y"], ub, uc)
    path = os.path.join(L.WORK, "reference_estimation.npz")
    np.savez(path, **ref)
    summary = lambda t: {"origins": int(len(t["row"])), "positive_origins": int(t["y"].sum()), "events": int(len(L.event_onsets(t["row"], t["y"]))),
                         "first": str(t["time"].min()), "last": str(t["time"].max())}
    out = {"_code": code, "baseline": L.BASELINES[best], "validation_detection_rates_alpha05": dict(zip(L.BASELINES, rates)),
           "coef_s": coef_s.tolist(), "coef_c": coef_c.tolist(), "reference_file": path, "reference_sha256": L.sha256(path),
           "tables": {k: summary(t) for k, t in tabs.items()}, "label_threshold": L.LABEL_THRESHOLD}
    save("frozen.json", out)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    stage = sys.argv[1]
    code = code_state()
    {"copy": stage_copy, "reproduce": stage_reproduce, "leak": stage_leak, "freeze": stage_freeze}[stage](code)
