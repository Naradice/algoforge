"""P3 RESUME driver (identical to p3_run.py except that existing ensembles are skipped).
P3 driver (plan v0.3 @ c65cede, section 7). Stops at the first failed gate.
Phase 0 (gates, diagnostic) is run separately (p3_gates.py, p3_diag.py) and checked here.
Phase 1a: C0s x 3 seeds -> C0s gate (9B). Phase 1b: C3, C1_60, C2_60, C5_60, C4_60 x 3 seeds.
Phase 2 (exploratory): C1/C2/C5 at B=15,240 with seed 1; further seeds only for conditions meeting the allocation guide."""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_env as E  # noqa: E402

PY = sys.executable


def run(*args):
    print(">>", *args, flush=True)
    r = subprocess.run([PY, os.path.join(HERE, args[0]), *args[1:]], cwd=HERE)
    return r.returncode


def gen(cond, seed):
    if os.path.exists(os.path.join(E.ENS_DIR, f"{cond}_seed{seed}.npz")):
        print("skip (exists)", cond, seed, flush=True)
        return
    rc = run("p3_gen.py", cond, str(seed))
    if rc != 0:
        raise SystemExit(f"generation failed: {cond} seed {seed}")


if __name__ == "__main__":
    E.code_state()
    g = json.load(open(os.path.join(E.OUT, "gates.json"), encoding="utf-8"))
    assert g["all_pass"], "phase-0 gates must pass"
    d = json.load(open(os.path.join(E.OUT, "diag_results.json"), encoding="utf-8"))   # phase-0 diagnostic must exist (reported, not a pass/fail gate)
    for s in (1, 2, 3):
        gen("C0s", s)
    if run("p3_c0s_gate.py") != 0:
        raise SystemExit("C0s consistency gate FAILED: stop and report (plan 9B)")
    for cond in ("C3", "C1_60", "C2_60", "C5_60", "C4_60"):
        for s in (1, 2, 3):
            gen(cond, s)
    print("PHASE 1 COMPLETE", flush=True)
    for cond in ("C1_15", "C1_240", "C2_15", "C2_240", "C5_15", "C5_240"):
        gen(cond, 1)
    print("PHASE 2 seed 1 COMPLETE", flush=True)
    if run("p3_flag.py") != 0:
        raise SystemExit("flagging failed")
    flagged = json.load(open(os.path.join(E.OUT, "phase2_flags.json")))["flagged"]
    for cond in flagged:
        for s in (2, 3):
            gen(cond, s)
    print("PHASE 2 COMPLETE (flagged:", flagged, ")", flush=True)
