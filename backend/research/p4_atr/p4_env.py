"""P4 environment: paths and code-state record (plan v0.2 @ 9d9825d). Results are written to OUT (outside the repo tree) so the p4_atr
code stays clean during a run; they are copied to p4_atr/results after the run."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(BACKEND, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "p3_burnin"))
sys.path.insert(0, os.path.join(HERE, "..", "p2_diagnostics"))
sys.path.insert(0, os.path.join(HERE, "..", "p1_verification"))

OUT = r"C:\tmp\p4_out"
P3_RESULTS = os.path.join(BACKEND, "research", "p3_burnin", "results")
PARQUET_SHA256 = "c25eb5720a5552b0230fb6ce2354de1cee020e6b7ca95da46034b75f2a53c828"   # recorded in the P3 gates (raw parquet hash)


def code_state(require_clean=True):
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "backend/research/p4_atr"], text=True).strip()
    if require_clean and dirty:
        raise SystemExit("p4_atr has uncommitted changes -> commit first: " + dirty)
    return {"head": head, "dirty_p4_files": dirty}
