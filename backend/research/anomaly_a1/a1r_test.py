"""End-to-end check of a1r.py on synthetic inputs only (no research data is read).  python a1r_test.py <empty scratch dir>"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

import a1_lib as L
import a1_data as D
import a1_s3 as S3
import a1r
from a1_test import series, write_zip


def expect_stop(fn, text):
    try:
        fn()
    except SystemExit as e:
        assert text in str(e), e
        return
    raise AssertionError("no stop: " + text)


def main(tmp):
    s = series("2021-09-01", "2022-12-30", 7)
    cut = pd.Timestamp("2022-08-31 17:59")
    old, new = s[s.index <= cut], s[(s.index > cut) & (s.index < "2022-11-01")]
    arch_dir, results, work = (os.path.join(tmp, x) for x in ("archives", "results", "work"))
    for d in (arch_dir, results, work):
        os.makedirs(d)
    ledger = {}
    keys = new.index.strftime("%Y-%m").where(new.index >= "2022-09-01", "2022-09")  # like the real files, 08-31 evening is in 2022-09
    for k, (month, g) in enumerate(new.groupby(keys)):
        p = os.path.join(arch_dir, f"ticks_USDJPY-oj5k_{month}.zip")
        write_zip(p, g, 10 + k)
        ledger[month] = D.ledger_entry(p)
    p8 = os.path.join(arch_dir, "ticks_USDJPY-oj5k_2022-08.zip")
    write_zip(p8, old["2022-08-01":], 9)
    months = ["2022-09", "2022-10"]
    # redirect every input and output to the scratch directory
    D.FORWARD_DIR, L.RESULTS, a1r.WORK = arch_dir, results, work
    L.load_main = lambda: (old.values.astype(float), old.index)
    L.f1_edges = lambda: list(np.linspace(0.001, 0.05, 19))
    a1r.PERIOD = ("2022-09-01", "2022-10-31")
    chk = D.check202208(p8)
    assert chk["identical"], chk
    chk["source_folder"] = arch_dir
    a1r.save("ledger.json", ledger)
    # stop conditions of the data stage
    a1r.save("check202208.json", {**chk, "identical": False})
    expect_stop(lambda: a1r.run_data({}, months), "2022-08")
    a1r.save("check202208.json", chk)
    expect_stop(lambda: a1r.run_data({}, months + ["2022-11"]), "missing")
    a1r.save("ledger.json", {**ledger, "2022-09": {**ledger["2022-09"], "sha256": "0"}})
    expect_stop(lambda: a1r.run_data({}, months), "differ")
    assert not os.path.exists(a1r.series_path())
    a1r.save("ledger.json", ledger)
    data = a1r.run_data({}, months)
    full = s[s.index < "2022-11-01"]
    z = np.load(a1r.series_path())
    assert data["passed"] and data["rows"] == len(full) and np.allclose(z["close"], full.values, atol=1e-9), data
    assert data["duplicated_minutes"] == 0 and data["minutes_not_after_main_series"] == 0 and data["weekly_open"]["share"] >= 0.95
    rng = np.random.default_rng(4)
    frozen = {"baseline": "rv24", "coef_s": [-3.0, 2.0, 0.5], "coef_c": [-3.0, 2.0, 0.5],
              "_ref": {k: np.sort(rng.random(500) * (1e-5 if k == "base" else 1)) for k in ("base", "S", "C")}}
    blocks = (("2022-09-01", "2022-09-30"), ("2022-10-01", "2022-10-31"))
    out = a1r.run_score({}, frozen, n_boot=100, n_leak=12, workers=3, blocks=blocks)
    pr = out["primary"]
    assert out["leak"]["passed"] and pr["origins"] > 300 and pr["events"] > 0 and out["retrospective_judgement"].startswith("後ろ向き"), out
    assert sum(b["origins"] for b in out["blocks"]) == pr["origins"] and sum(b["positive_origins"] for b in out["blocks"]) == pr["positive_origins"]
    assert all("ci_delta" not in b and "judgement" not in b for b in out["blocks"])
    assert not os.path.exists(os.path.join(results, "evaluation.json"))
    expect_stop(lambda: a1r.run_score({}, frozen), "scored once")
    assert S3.month_list("2022-09", "2026-10") == a1r.MONTHS and len(a1r.MONTHS) == 50 and a1r.MONTHS[-1] < "2026-12"
    print("A1R end-to-end check passed:", json.dumps({k: pr[k] for k in ("origins", "events", "alerts")}))


if __name__ == "__main__":
    main(sys.argv[1])
