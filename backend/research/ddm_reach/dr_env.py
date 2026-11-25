"""DDM reachable-region study environment (plan v0.6 @ 0c1cbce, approved). Outputs: OUT (outside the repo) and results/ in the repo."""
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

OUT = r"C:\tmp\ddm_reach_out"
A1_DIR = os.path.join(OUT, "a1")
RESULTS = os.path.join(HERE, "results")
ATLAS = os.path.join(BACKEND, "ddm_regime_atlas.csv")
PARQUET = os.path.join(REPO, "artifacts", "datasets", "derived", "usdjpy_1min_volatility.parquet")
PARQUET_SHA = "c25eb5720a5552b0230fb6ce2354de1cee020e6b7ca95da46034b75f2a53c828"
BANDS = os.path.join(BACKEND, "research", "p5b_stage3", "results", "bands.json")
MAPPING = r"C:\tmp\f1_run1_db0926f\ddm_mapping.json"
PLAN_COMMIT = "0c1cbce"

# ---- A1 grid (plan 4.2-4.3, fixed) -------------------------------------------------------------------------------------------
DELTAS = [round(-1.5 + 0.1 * i, 1) for i in range(16)]
WMAS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40]
N_POINTS = len(DELTAS) * len(WMAS)              # 176
SEEDS_PER_SET = 5
SETS = {"SA": 1, "SB": 2}
N_TRADES = 100_001                              # closes = price[0::TPC]; TPC=20 -> 5001 closes -> 5 windows of 1000 returns
NUM_AGENT = 300
TICK = 0.001
TPCS = (10, 20, 40)
TPC_MAIN = 20
L_MAIN, L_SENS = 1000, 4000
WORKERS = 18

M_NAMES = ("acf_1", "kurtosis", "volclust_50")                    # PA, matching
E_NAMES = ("acf_5", "acf_20", "vr20", "vr60", "skew")             # PA, evaluation


def point_id(i_delta, i_wma):
    return i_delta * len(WMAS) + i_wma


def grid():
    return [(point_id(i, j), DELTAS[i], WMAS[j]) for i in range(len(DELTAS)) for j in range(len(WMAS))]


def seed_of(set_name, pid, k):
    return 100_000 * SETS[set_name] + pid * 10 + k


def run_file(set_name, pid, k):
    return os.path.join(A1_DIR, f"ticks_{set_name}_{pid:03d}_{k}.npy")


def code_state(require_clean=True):
    head = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", "backend/research/ddm_reach"], text=True).strip()
    if require_clean and dirty:
        raise SystemExit("ddm_reach has uncommitted changes -> commit first: " + dirty)
    return {"head": head, "dirty_ddm_reach_files": dirty}
