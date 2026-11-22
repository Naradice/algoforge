"""P5-B stage 2 environment (plan v0.2 @ 6d6af4c). Regenerated closes and statistics go to OUT (outside the repo tree)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(BACKEND, ".."))
for sub in ("p5b_scale", "p5a_short", "p4_atr", "p3_burnin", "p2_diagnostics", "p1_verification"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
sys.path.insert(0, BACKEND)

OUT = r"C:\tmp\p5b2_out"
ENS_DIR = r"C:\tmp\p3_out\ens"
RUN_DIR = r"C:\tmp\f1_run1_db0926f"
STAGE1_RESULTS = os.path.join(BACKEND, "research", "p5b_scale", "results")
BUCKETS = (0, 8, 15, 19)
CONDS = ("C3", "C0s")
SEED_ID = 1
N_PATHS = 1000
N_CAND = 8
WORKERS = 18
PAR_ATTEMPTS = (0, 127, 254, 381, 508, 635, 762, 889)       # parallel-independence check: 8 attempts per (condition, bucket)
P3_COMMIT = "a113456771683cad888518fcba75cffc6566f206"       # code state of the P3 ensemble generation
F1_COMMIT = "db0926f1afd6353339a90abeeac9717a400afe4b"


def code_state(require_clean=True):
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "backend/research/p5b_stage2"], text=True).strip()
    if require_clean and dirty:
        raise SystemExit("p5b_stage2 has uncommitted changes -> commit first: " + dirty)
    return {"head": head, "dirty_p5b_stage2_files": dirty}
