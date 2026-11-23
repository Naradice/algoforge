"""P5-B stage 3 environment (plan v0.3 @ 496ea6d). Stored data only; no simulation. Outputs: OUT (outside the repo) and results/ in the repo."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(BACKEND, ".."))
for sub in ("p5b_stage2", "p5b_scale", "p5a_short", "p4_atr", "p3_burnin", "p2_diagnostics", "p1_verification"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
sys.path.insert(0, BACKEND)

OUT = r"C:\tmp\p5b3_out"
S2_OUT = r"C:\tmp\p5b2_out"
RUN_DIR = r"C:\tmp\f1_run1_db0926f"
ATLAS = os.path.join(BACKEND, "ddm_regime_atlas.csv")
S2_RESULTS = os.path.join(BACKEND, "research", "p5b_stage2", "results")
RESULTS = os.path.join(HERE, "results")
BUCKETS4 = (0, 8, 15, 19)
CONDS = ("C3", "C0s")
N_CAND = 8
WORKERS = 12
LOG125 = 0.22314355131420976   # log(1.25)
BAND_FILE = os.path.join(RESULTS, "bands.json")


def code_state(require_clean=True):
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "backend/research/p5b_stage3"], text=True).strip()
    if require_clean and dirty:
        raise SystemExit("p5b_stage3 has uncommitted changes -> commit first: " + dirty)
    return {"head": head, "dirty_p5b_stage3_files": dirty}


def load_mapping():
    return json.load(open(os.path.join(RUN_DIR, "ddm_mapping.json")))
