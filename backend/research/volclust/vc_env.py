"""volclust_50 verification, revision r1 (plan docs/research-ddm-volclust-verification-plan-r1.md v0.2 @ 9f067ee, approved; everything it does not change follows
docs/research-ddm-volclust-verification-plan.md v0.3 @ c9b2a79). The code of the original run is at commits b45fb93 / d9607c8; its results/ folder is read only.
Constants, the shared wall-clock deadline (2 h from the first execution of R0-R4, never reset), the ledger and the output location.
Set VC_SYNTH=<dir> to dry-run every script on synthetic inputs (no research data is read, no deadline)."""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DR = os.path.abspath(os.path.join(HERE, "..", "ddm_reach"))
sys.path.insert(0, HERE)
sys.path.insert(0, DR)
import dr_env as DE  # noqa: E402  (also puts backend and the earlier research folders on sys.path)

REPO = DE.REPO
SYNTH = os.environ.get("VC_SYNTH")
RESULTS = os.path.join(SYNTH, "results") if SYNTH else os.path.join(HERE, "results_r1")
PREVIOUS = os.path.join(SYNTH, "previous") if SYNTH else os.path.join(HERE, "results")      # original run: read only
OUT = os.path.join(SYNTH, "cache") if SYNTH else r"C:\tmp\volclust_r1_out"
HIST_DIR = r"L:\data\fx"
A1_STATS = os.path.join(DE.OUT, "a1_stats.pkl")
A1_STATS_SHA = "e9e0bcdd9995504953557b7aea196df511e559a7c57a33b2a1c7e04f28faa6a8"
DR_RESULTS = os.path.join(SYNTH, "dr_results") if SYNTH else os.path.join(DR, "results")

LAG = 50
L_MAIN, L_SENS = 1000, 4000
DELTA = 0.022
DELTAS_SENS = (0.010, 0.030)
INDEP = ("2010-01-01", "2016-08-30")
SECOND = ("2016-08-31", "2020-08-31")
ESTIM = ("2016-08-31", "2020-08-31")
VALID = ("2020-09-01", "2021-08-31")
PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "EURJPY", "GBPJPY", "AUDJPY", "CHFJPY")
GAP_SECONDS, ZERO_FRAC_MAX, WEEK_SHIFT_HOURS = 300, 0.30, 6
MIN_BIN_OBS = 100
B = int(os.environ.get("VC_B", 2000))
B_CAL = int(os.environ.get("VC_BCAL", 500))
N_CAL = int(os.environ.get("VC_NCAL", 1000))
SOURCE_MIN_RETURNS = 950
SOURCE_BIAS_LIMIT = DELTA / 4                        # r1 3.2: the 90% interval of mean(hist - main) must lie within +-this, in estimation and validation
SOURCE_MIN_MATCHED, SOURCE_MIN_FRACTION = 250, 0.80  # r1 3.2: matched windows per period
OPEN_MINUTES, OPEN_CONFORM = 30, 0.95                # r1 2.3 (a): converted weekly open Monday 00:00-00:29 in >= 95% of weeks
CORR_MIN_MATCHED, CORR_CONFORM, CORR_MEDIAN = 500, 0.98, 0.90      # r1 2.3 (b)
CORR_MIN_SHIFTED = 100                               # a +-1 h correlation with fewer matched returns cannot count against offset 0
LIMIT_SECONDS = 7200
SEEDS = {"cal": 20261201, "real": 20261202, "ddm": 20261203, "inject": 20261204, "v3": 20261205, "source": 20261206}

_state = {}


def _write(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_conv)


def _conv(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    _write(os.path.join(RESULTS, name), obj)


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def load_previous(name):
    with open(os.path.join(PREVIOUS, name), encoding="utf-8") as f:
        return json.load(f)


def code_state():
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    if SYNTH:
        return {"head": head, "synthetic": True}
    dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "backend/research/volclust", ":(exclude)backend/research/volclust/results_r1",
                                     "backend/research/ddm_reach"], text=True).strip()
    if dirty:
        raise SystemExit("uncommitted code changes -> commit first: " + dirty)
    return {"head": head}


def _ledger(entry):
    p = os.path.join(RESULTS, "budget_ledger.json")
    rows = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []
    rows.append(entry)
    _write(p, rows)


def _finish(status):
    if _state.get("done"):
        return
    _state["done"] = True
    now = time.time()
    _ledger({"script": _state["name"], "start": _state["start"], "end": now, "seconds": now - _state["start"], "status": status,
             "seconds_since_first_execution": now - _state["t0"]})


def _kill_tree():
    try:
        import psutil
        for c in psutil.Process().children(recursive=True):
            c.kill()
    except Exception:
        pass
    _finish("killed at the shared deadline")
    print("DEADLINE exceeded: stopped and to be reported", flush=True)
    os._exit(3)


def begin(name, first=False):
    """checks the code state and the shared deadline, starts the watchdog; the first script creates results/deadline.json (never overwritten)."""
    code = code_state()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(RESULTS, "deadline.json")
    now = time.time()
    if not os.path.exists(p):
        if not first:
            raise SystemExit("no deadline.json: the first script of the run must start first")
        _write(p, {"first_execution_start_unix": now, "deadline_unix": now + LIMIT_SECONDS, "first_execution_start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                   "deadline": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now + LIMIT_SECONDS)), "first_script": name, "code_head": code["head"]})
    d = json.load(open(p, encoding="utf-8"))
    remaining = d["deadline_unix"] - now
    _state.update(name=name, start=now, t0=d["first_execution_start_unix"], done=False)
    if SYNTH:
        remaining = 10 * LIMIT_SECONDS
    if remaining <= 0:
        _finish("not started: deadline passed")
        raise SystemExit("DEADLINE passed: not started")
    t = threading.Timer(remaining, _kill_tree)
    t.daemon = True
    t.start()
    atexit.register(lambda: _finish(_state.get("status", "ended without completion")))
    return code


def done():
    _state["status"] = "completed"
