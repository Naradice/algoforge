"""P2 execution environment: paths and code-state record (aborts when p2_diagnostics has uncommitted changes)."""
import os
import subprocess

RUN_DIR = r"C:\tmp\f1_run1_db0926f"
SHA_LIST = r"C:\tmp\f1_sha256_58files.txt"
PARQUET = r"C:\Users\ksato\workspace\finance\algoforge\artifacts\datasets\derived\usdjpy_1min_volatility.parquet"
PARQUET_SHA = "c25eb5720a5552b0230fb6ce2354de1cee020e6b7ca95da46034b75f2a53c828"
OUT = r"C:\tmp\p2_out"
P1_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "p1_verification")


def code_state(require_clean=True):
    here = os.path.dirname(os.path.abspath(__file__))
    root = subprocess.check_output(["git", "-C", here, "rev-parse", "--show-toplevel"], text=True).strip()
    head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--", "backend/research/p2_diagnostics"], text=True).strip()
    if require_clean and dirty:
        raise SystemExit("p2_diagnostics has uncommitted changes -> commit first: " + dirty)
    return {"head": head, "dirty_p2_files": dirty}
