"""P5-B environment (plan v0.2 @ 5277e17). Results go to OUT (outside the repo tree) and are copied to p5b_scale/results after the run."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(BACKEND, ".."))
for sub in ("p5a_short", "p4_atr", "p3_burnin", "p2_diagnostics", "p1_verification"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
sys.path.insert(0, BACKEND)

OUT = r"C:\tmp\p5b_out"
RUN_DIR = r"C:\tmp\f1_run1_db0926f"
ENS_DIR = r"C:\tmp\p3_out\ens"
ATLAS = os.path.join(BACKEND, "ddm_regime_atlas.csv")
P5A_RESULTS = os.path.join(BACKEND, "research", "p5a_short", "results")


def code_state(require_clean=True):
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "backend/research/p5b_scale"], text=True).strip()
    if require_clean and dirty:
        raise SystemExit("p5b_scale has uncommitted changes -> commit first: " + dirty)
    return {"head": head, "dirty_p5b_files": dirty}
