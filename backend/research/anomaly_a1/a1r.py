"""A1R (Study Manager #1148), frozen plan docs/research-anomaly-a1r-plan.md v0.1: retrospective scoring of the frozen A1 settings on
2022-09-01..2026-10-31. Definitions and procedures are those of a1_lib / a1_s3; only the period, the months and the output places differ.

  python a1r.py data     R2: data checks and series construction (reads prices; computes no label and no score)
  python a1r.py score    R3: leak test, then ONE scoring, then the descriptive breakdown (refused when a result already exists)
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import a1_lib as L
import a1_data as D
import a1_s3 as S3
fm, fc = L.fm, L.fc

PERIOD = ("2022-09-01", "2026-10-31")
FIRST_MONDAY = "2022-09-05"
MONTHS = S3.month_list("2022-09", "2026-10")
BLOCKS = (("2022-09-01", "2023-08-31"), ("2023-09-01", "2024-08-31"), ("2024-09-01", "2025-08-31"), ("2025-09-01", "2026-10-31"))
WORK = os.path.join(L.REPO, "artifacts", "research", "anomaly_a1r")
LABELS = {"支持": "後ろ向きの支持", "上乗せなし": "後ろ向きの上乗せなし", "判定保留": "後ろ向きの保留"}


def result_path(name):
    return os.path.join(L.RESULTS, name)


def save(name, obj):
    json.dump(obj, open(result_path(name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def load(name):
    return json.load(open(result_path(name), encoding="utf-8"))


def series_path():
    return os.path.join(WORK, "series_retro.npz")


def code_state():
    """as for A1 (committed code, unchanged F1 code, same library versions), plus a committed A1R plan."""
    import subprocess
    from a1_s1 import code_state as a1_code_state
    out = a1_code_state()
    dirty = subprocess.check_output(["git", "-C", L.REPO, "status", "--porcelain", "--", "docs/research-anomaly-a1r-plan.md"], text=True).strip()
    if dirty:
        raise SystemExit("uncommitted A1R plan:\n" + dirty)
    return out


def stop(name, res, why):
    res["passed"] = False
    res["stopped"] = why
    save(name, res)
    raise SystemExit("STOP: " + why)


def run_data(code, months=MONTHS):
    """plan section 5, R2."""
    res = {"_code": code, "period": list(PERIOD), "months": len(months), "passed": False}
    arch, ledger, chk = D.archives(D.FORWARD_DIR), load("ledger.json"), load("check202208.json")
    # 1. the re-downloaded 2022-08 reproduces the main series
    if not (chk["identical"] and "2022-08" in arch and L.sha256(arch["2022-08"]) == chk["sha256"]
            and os.path.normcase(chk["source_folder"]) == os.path.normcase(D.FORWARD_DIR)):
        stop("retro_data.json", res, "the check of the re-downloaded 2022-08 archive did not pass")
    res["check202208"] = {k: chk[k] for k in ("sha256", "rows_archive", "identical")}
    # 2. every month present and identical to the ledger
    missing = [m for m in months if m not in arch or m not in ledger]
    if missing:
        stop("retro_data.json", res, f"missing archives or ledger entries: {missing}")
    differ = [m for m in months if L.sha256(arch[m]) != ledger[m]["sha256"]]
    if differ:
        stop("retro_data.json", res, f"archives differ from the ledger: {differ}")
    res["backward_tick_steps"] = {m: ledger[m]["backward_steps"] for m in months if ledger[m]["backward_steps"]}
    # 3. 1-minute closes: no duplicated or backward minute, nothing at or before the end of the main series
    close, index = L.load_main()
    with ProcessPoolExecutor(max_workers=6) as ex:
        minutes = list(ex.map(D.minute_closes, [arch[m] for m in months]))
    new = pd.DatetimeIndex(np.concatenate([x.index.values for x in minutes]))
    res["new_minutes"] = int(len(new))
    res["duplicated_minutes"] = int(new.duplicated().sum())
    res["backward_minutes"] = int((np.diff(new.values).astype("int64") < 0).sum())
    res["minutes_not_after_main_series"] = int((new <= index.max()).sum())
    if res["duplicated_minutes"] or res["backward_minutes"]:
        stop("retro_data.json", res, "duplicated or backward minutes")
    n_main = len(close)
    close, index = S3.extend_series(close, index, minutes)
    bad = int((~np.isfinite(close) | (close <= 0)).sum())
    res.update({"main_rows": int(n_main), "rows": int(len(close)), "last": str(index.max()), "non_positive_or_non_finite_closes": bad})
    assert index.max() < pd.Timestamp(PERIOD[1]) + pd.Timedelta(days=1), "rows after the period"
    if bad:
        stop("retro_data.json", res, "invalid closes")
    # 4. weekly open
    res["weekly_open"] = S3.weekly_open(index, FIRST_MONDAY, PERIOD[1])
    if res["weekly_open"]["share"] < S3.OPEN_SHARE:
        stop("retro_data.json", res, "weekly open check failed")
    os.makedirs(WORK, exist_ok=True)
    np.savez(series_path(), close=close, index=index.values.astype("datetime64[ns]").astype("int64"), edges=np.asarray(L.f1_edges()))
    res.update({"series_file": series_path(), "series_sha256": L.sha256(series_path()), "passed": True})
    save("retro_data.json", res)
    return res


def breakdown(table_file, ctx, frozen, blocks):
    """descriptive point estimates per block; alerts (and event onsets) are formed from the block's own origins."""
    z = dict(np.load(table_file))
    z["time"] = ctx.tt[z["j"]]
    out = []
    for a, b in blocks:
        m = np.asarray((z["time"] >= pd.Timestamp(a)) & (z["time"] < pd.Timestamp(b) + pd.Timedelta(days=1)))
        r = L.compare({k: v[m] for k, v in z.items()}, frozen, L.ALPHA, 0)
        out.append({"from": a, "to": b, **r})
    return out


def run_score(code, frozen, n_boot=L.BOOT, n_leak=L.LEAK_N, workers=S3.WORKERS, blocks=BLOCKS):
    """plan section 5, R3. The scoring result is written before the breakdown is computed."""
    for name in ("retro_evaluation.json", "retro_evaluation_stopped.json"):
        if os.path.exists(result_path(name)):
            raise SystemExit(f"{name} exists; the period is scored once")
    data = load("retro_data.json")
    assert data["passed"] and L.sha256(series_path()) == data["series_sha256"], "the data checks have not passed for this series"
    start = time.time()
    fc.PERIODS["forward"] = PERIOD                                              # a1_s3.score_period scores this period name
    res = S3.score_period(series_path(), frozen, n_boot=n_boot, n_leak=n_leak, workers=workers, deadline=start + S3.TIME_LIMIT)
    res = {"_code": code, "period": list(PERIOD), "series_sha256": data["series_sha256"], "seconds": time.time() - start, **res}
    if "primary" not in res:
        save("retro_evaluation_stopped.json", res)
        raise SystemExit("STOP: " + res["stopped"])
    res["retrospective_judgement"] = LABELS[res["primary"]["judgement"]]
    save("retro_evaluation.json", res)
    ctx = S3._context(series_path())
    res["blocks"] = breakdown(os.path.join(WORK, "evaluation_table.npz"), ctx, frozen, blocks)
    res["seconds"] = time.time() - start
    save("retro_evaluation.json", res)
    return res


def main():
    stage = sys.argv[1]
    code = code_state()
    if stage == "data":
        out = run_data(code)
    elif stage == "score":
        out = run_score(code, S3.load_frozen())
    else:
        raise SystemExit(__doc__)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
