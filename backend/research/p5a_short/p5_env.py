"""P5-A environment (plan v0.2 @ b995b5a). Results are written to OUT (outside the repo tree); copied to p5a_short/results after the run."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(BACKEND, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "p4_atr"))
sys.path.insert(0, os.path.join(HERE, "..", "p3_burnin"))
sys.path.insert(0, os.path.join(HERE, "..", "p2_diagnostics"))
sys.path.insert(0, os.path.join(HERE, "..", "p1_verification"))

OUT = r"C:\tmp\p5a_out"
RUN_DIR = r"C:\tmp\f1_run1_db0926f"
ENS_DIR = r"C:\tmp\p3_out\ens"
P4_RESULTS = os.path.join(BACKEND, "research", "p4_atr", "results")
P3_RESULTS = os.path.join(BACKEND, "research", "p3_burnin", "results")


def code_state(require_clean=True):
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "backend/research/p5a_short"], text=True).strip()
    if require_clean and dirty:
        raise SystemExit("p5a_short has uncommitted changes -> commit first: " + dirty)
    return {"head": head, "dirty_p5a_files": dirty}
