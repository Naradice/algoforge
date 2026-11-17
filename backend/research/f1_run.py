"""F1 (#889 v0.3): run all models on a period, score per origin, and run the leak test.

  python research/f1_run.py score <workdir> <period>      period in {validation, final}
  python research/f1_run.py leak  <workdir>
Requires <workdir>/ddm_mapping.json and ddm_ens_seed{1,2,3}.npy (from f1_ddm.py).
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import f1_common as fc  # noqa: E402
import f1_models as fm  # noqa: E402

STAT_MODELS = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7N", "B7T", "B8", "B4s"]
N_DDM_SEEDS = 3
LEAK_ORIGINS = 150


def load_ctx(workdir):
    mp = json.load(open(os.path.join(workdir, "ddm_mapping.json")))
    df = pd.read_parquet(fc.artifact_path(), columns=["close"])
    return fm.Context(df["close"].values, df.index, mp["edges"]), df


def load_ens(workdir):
    return [np.load(os.path.join(workdir, f"ddm_ens_seed{s}.npy")) for s in range(1, N_DDM_SEEDS + 1)]


def ddm_samples(ens_list, ctx, j, seed_i):
    b = int(ctx.bucket[j])
    return ens_list[seed_i][b]                              # [4, 1000]


def month_job(args):
    workdir, period, month_start = args
    ctx, _ = load_ctx(workdir)
    ens = load_ens(workdir)
    st = fm.build_state(ctx, month_start, None)
    m0 = month_start
    m1 = month_start + pd.offsets.MonthBegin(1)
    in_month = np.asarray((ctx.tt >= m0) & (ctx.tt < m1)) & ctx.origin_ok & ctx.s.period_mask(period)
    origins = np.flatnonzero(in_month)
    valid_h = {h: ctx.s.period_mask(period, h) & np.isfinite(ctx.targets[h]) for h in fc.HORIZONS}
    origins = np.array([j for j in origins if any(valid_h[h][j] for h in fc.HORIZONS)])
    out = {}
    fails = list(st.arch_fail)
    models = STAT_MODELS + [f"DDM{i + 1}" for i in range(N_DDM_SEEDS)]
    for name in models:
        samp = {h: [] for h in fc.HORIZONS}
        js = {h: [] for h in fc.HORIZONS}
        for j in origins:
            if name.startswith("DDM"):
                x = ddm_samples(ens, ctx, j, int(name[3:]) - 1)
            else:
                x = fm.SAMPLERS[name](st, ctx, j)
            if x is None or not np.all(np.isfinite(x)):
                fails.append((name, int(j)))
                continue
            for hi, h in enumerate(fc.HORIZONS):
                if valid_h[h][j]:
                    samp[h].append(x[hi])
                    js[h].append(j)
        for h in fc.HORIZONS:
            if not js[h]:
                continue
            y = ctx.targets[h][np.array(js[h])]
            rng = np.random.default_rng([99, fm.MODEL_IDS.get(name, 20 + int(name[3:]) if name.startswith("DDM") else 0), h, month_start.month, month_start.year])
            sc = fc.score_samples(np.array(samp[h]), y, rng)
            sc["j"] = np.array(js[h])
            out[(name, h)] = sc
    return str(month_start.date()), out, fails


def stage_score(workdir, period):
    months = fm.month_starts(period)
    t0 = time.time()
    results = {}
    fails = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        for mstr, out, f in ex.map(month_job, [(workdir, period, m) for m in months]):
            fails[mstr] = f
            for k, sc in out.items():
                results.setdefault(k, []).append(sc)
            print(f"{period} {mstr} done ({time.time() - t0:.0f}s) failures={len(f)}", flush=True)
    ctx, _ = load_ctx(workdir)
    save = {}
    for (name, h), parts in results.items():
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        order = np.argsort(cat["j"])
        for k, v in cat.items():
            save[f"{name}|h{h}|{k}"] = v[order]
    np.savez(os.path.join(workdir, f"scores_{period}.npz"), **save)
    json.dump({k: v for k, v in fails.items() if v}, open(os.path.join(workdir, f"failures_{period}.json"), "w"), default=str)
    # headline
    print("mean fair CRPS (x1e4):")
    for name in STAT_MODELS + [f"DDM{i + 1}" for i in range(N_DDM_SEEDS)]:
        row = []
        for h in fc.HORIZONS:
            key = f"{name}|h{h}|crps"
            row.append(f"{save[key].mean() * 1e4:8.4f}" if key in save else "   n/a  ")
        print(f"  {name:5s}", " ".join(row))
    n_origins = {h: int(len(save[f'B1|h{h}|j'])) for h in fc.HORIZONS}
    print("origins per horizon:", n_origins)


# ----------------------------------------------------------------------------- leak test
def leak_job(args):
    workdir, period, seed = args
    ctx, df = load_ctx(workdir)
    ens = load_ens(workdir)
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(ctx.origin_ok & ctx.s.period_mask(period) & np.isfinite(ctx.targets[1]))
    picks = rng.choice(cand, size=min(LEAK_ORIGINS, len(cand)), replace=False)
    mp = json.load(open(os.path.join(workdir, "ddm_mapping.json")))
    states = {}
    bad = []
    close = df["close"].values
    for j in picks:
        ms = pd.Timestamp(ctx.tt[j].year, ctx.tt[j].month, 1)
        if ms not in states:
            states[ms] = fm.build_state(ctx, ms, None)
        st = states[ms]
        trow = int(ctx.s.t[j])
        pert = close.copy()
        steps = rng.normal(0, 0.002, size=len(close) - trow - 1)
        pert[trow + 1:] = close[trow] * np.exp(np.cumsum(steps))
        ctx_p = fm.Context(pert, df.index, mp["edges"])
        for name in STAT_MODELS:
            a = fm.SAMPLERS[name](st, ctx, j)
            b = fm.SAMPLERS[name](st, ctx_p, j)
            same = (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b))
            if not same:
                bad.append((name, int(j)))
        for si in range(N_DDM_SEEDS):
            if not np.array_equal(ddm_samples(ens, ctx, j, si), ddm_samples(ens, ctx_p, j, si)):
                bad.append((f"DDM{si + 1}", int(j)))
    return period, len(picks), bad


def stage_leak(workdir):
    jobs = [(workdir, p, 7000 + i) for i, p in enumerate(("validation", "final"))]
    with ProcessPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(leak_job, jobs))
    ok = True
    for period, n, bad in res:
        print(f"leak test {period}: {n} origins x {len(STAT_MODELS) + N_DDM_SEEDS} models, mismatches={len(bad)}")
        ok &= (len(bad) == 0)
        if bad:
            print("  ", bad[:10])
    json.dump({"pass": ok, "results": [{"period": p, "n": n, "mismatches": b} for p, n, b in res]},
              open(os.path.join(workdir, "leak_test.json"), "w"), indent=1)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "score":
        stage_score(sys.argv[2], sys.argv[3])
    else:
        stage_leak(sys.argv[2])
