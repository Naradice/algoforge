"""Revision r1: runs R0, R1, V1 (R2), V2 (R3), V3a / V3b (R4) in order, committing the results after every step (the adjusted rules are committed before V3b
compares). Stops on a failed step and at the deadline; the gates of R0 and R1 are applied inside the following steps."""
from __future__ import annotations

import subprocess
import sys

import vc_env as E

STEPS = ("vc_r0", "vc_r1", "vc_v1", "vc_v2", "vc_v3a", "vc_v3b")


def commit(msg):
    if E.SYNTH:
        return
    subprocess.check_call(["git", "-C", E.REPO, "add", "--", "backend/research/volclust/results_r1"])
    subprocess.check_call(["git", "-C", E.REPO, "commit", "-q", "-m", msg, "-m", "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"])


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else STEPS[0]
    for step in STEPS[STEPS.index(start):]:
        print(f"== {step}", flush=True)
        rc = subprocess.call([sys.executable, f"{E.HERE}/{step}.py"])
        commit(f"volclust verification r1: results of {step} (exit {rc})")
        if rc != 0:
            print(f"stopped: {step} exit {rc}", flush=True)
            sys.exit(rc)
    print("R0-R4 completed", flush=True)


if __name__ == "__main__":
    main()
