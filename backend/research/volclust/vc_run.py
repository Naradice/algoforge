"""Runs V0-V3 in order, committing the results after every step (frozen rules are committed before V3b compares). Stops on a failed step, on the V0 gate and at the deadline."""
from __future__ import annotations

import subprocess
import sys

import vc_env as E

STEPS = ("vc_v0a", "vc_v0b", "vc_v0c", "vc_v1", "vc_v2", "vc_v3a", "vc_v3b")


def commit(msg):
    if E.SYNTH:
        return
    subprocess.check_call(["git", "-C", E.REPO, "add", "--", "backend/research/volclust/results"])
    subprocess.check_call(["git", "-C", E.REPO, "commit", "-q", "-m", msg, "-m", "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"])


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else STEPS[0]
    for step in STEPS[STEPS.index(start):]:
        print(f"== {step}", flush=True)
        rc = subprocess.call([sys.executable, f"{E.HERE}/{step}.py"])
        commit(f"volclust verification: results of {step} (exit {rc})")
        if rc != 0:
            print(f"stopped: {step} exit {rc}", flush=True)
            sys.exit(rc)
        if step == "vc_v0c" and E.load("v0c.json")["stop_false_diff"]:
            print("stopped: V0 false 'difference' rate above 10% (rule not changed)", flush=True)
            sys.exit(4)
    print("V0-V3 completed", flush=True)


if __name__ == "__main__":
    main()
