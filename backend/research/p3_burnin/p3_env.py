"""P3 environment: paths, constants, code-state record (plan v0.3 @ c65cede)."""
import hashlib
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(BACKEND, ".."))
RUN_DIR = r"C:\tmp\f1_run1_db0926f"
OUT = r"C:\tmp\p3_out"
ENS_DIR = os.path.join(OUT, "ens")
F1_COMMIT = "db0926f1afd6353339a90abeeac9717a400afe4b"
SIM_REL = "backend/data/collectors/ddm_simulator.py"
F1DDM_REL = "backend/research/f1_ddm.py"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def code_state(require_clean=True):
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "backend/research/p3_burnin"], text=True).strip()
    if require_clean and dirty:
        raise SystemExit("p3_burnin has uncommitted changes -> commit first: " + dirty)
    return {"head": head, "dirty_p3_files": dirty}


def simulator_identity():
    """working-tree simulator and f1_ddm.py must be identical (modulo line endings) to the F1 run commit."""
    out = {}
    for rel in (SIM_REL, F1DDM_REL):
        committed = subprocess.check_output(["git", "-C", REPO, "show", f"{F1_COMMIT}:{rel}"])
        with open(os.path.join(REPO, rel), "rb") as f:
            cur = f.read()
        a = committed.replace(b"\r\n", b"\n")
        b = cur.replace(b"\r\n", b"\n")
        out[rel] = {"sha256_f1_commit": sha256_bytes(a), "sha256_working": sha256_bytes(b), "identical": a == b}
    return out
