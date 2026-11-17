"""F1 (#889): run guard. Every executable stage calls `start_run` first and `end_run` last.

Division of responsibility
  * This module MEASURES the real state (HEAD, every repository module loaded by the run, tracked-and-clean status, input
    hashes, package versions) and ENFORCES a negative gate decision (it refuses to run).
  * The Study Manager DECIDES (`check_execution_gate`: plan approved, Stage C approved, no unresolved critical, audited commit
    equals the target commit, clean workspace).

Procedure (no environment variable and no flag can bypass it)
  1. `python research/f1_run.py score <workdir> validation --preflight` (any stage script)  -> prints HEAD, dirty files and the `state_hash`
     (sha256 over HEAD + the hash of every loaded repository module + the input hashes) and writes the preflight manifest.
  2. Call the Study Manager `check_execution_gate(node_id, target_commit_sha=HEAD, target_dirty=<dirty?>)` with exactly the
     printed values and save the response as `<workdir>/gate_approval_<stage>.json`:
       {"node_id": ..., "target_commit_sha": ..., "state_hash": ..., "gate_response": <the response>}
  3. Run the stage (each stage script also accepts `--preflight` to print the same state without running). `start_run` recomputes the state, refuses if any loaded repository module is modified or untracked, and
     refuses unless the approval file belongs to this node, this HEAD and this exact state hash and the gate response says
     passed / stage_c_approved / commit_match / clean_workspace.
  4. `end_run` re-measures after the run (lazily imported modules) and fails the run if anything differs from HEAD.
The approval file is evidence written by the caller, so the Study Manager record of the same gate call is the authority in
the post-run audit; the guard does not make the file unforgeable.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

NODE_ID = 889


def _git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check).stdout


def repo_root():
    return Path(_git(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel").strip()).resolve()


def sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def loaded_repo_files(repo):
    """Source files of every module currently imported from inside the repository."""
    repo = Path(repo).resolve()
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not str(f).endswith(".py"):
            continue
        p = Path(f).resolve()
        try:
            p.relative_to(repo)
        except ValueError:
            continue
        out.add(p)
    return sorted(out)


def _dirty(repo, files):
    """Files that are modified or untracked relative to HEAD (git decides, so line-ending normalisation is respected)."""
    rels = [str(Path(f).resolve().relative_to(Path(repo).resolve())) for f in files]
    if not rels:
        return []
    lines = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *rels).splitlines()
    return sorted(ln[3:].strip() for ln in lines)


def measure_state(repo, inputs=(), preload=()):
    repo = Path(repo).resolve()
    for name in preload:                                   # modules that are imported lazily inside functions
        importlib.import_module(name)
    files = loaded_repo_files(repo)
    input_files = [Path(p).resolve() for p in inputs if Path(p).exists()]
    tracked_inputs = []
    for p in input_files:
        try:
            p.relative_to(repo)
        except ValueError:
            continue
        if subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", str(p)], capture_output=True).returncode == 0:
            tracked_inputs.append(p)
    head = _git(repo, "rev-parse", "HEAD").strip()
    code_hashes = {str(p.relative_to(repo)).replace("\\", "/"): sha256(p) for p in files}
    input_hashes = {str(p): sha256(p) for p in input_files}
    state = {"head": head, "code": code_hashes, "inputs": input_hashes,
             "dirty": _dirty(repo, files + tracked_inputs)}
    state["state_hash"] = hashlib.sha256(json.dumps({k: state[k] for k in ("head", "code", "inputs")}, sort_keys=True).encode()).hexdigest()
    return state


def _manifest(workdir, stage, kind, state, argv, extra=None):
    m = {"stage": stage, "kind": kind, "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "argv": list(argv),
         "python": sys.version.split()[0], **state}
    for mod in ("numpy", "scipy", "pandas", "arch", "statsmodels"):
        try:
            m[mod] = sys.modules[mod].__version__ if mod in sys.modules else __import__(mod).__version__
        except Exception:
            m[mod] = None
    if extra:
        m.update(extra)
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, f"run_manifest_{stage}_{kind}_{m['utc'][:19].replace(':', '')}.json")
    with open(path, "w") as f:
        json.dump(m, f, indent=1)
    return path


def preflight(workdir, stage, repo=None, inputs=(), preload=(), argv=None):
    repo = repo or repo_root()
    state = measure_state(repo, inputs, preload)
    _manifest(workdir, stage, "preflight", state, argv if argv is not None else sys.argv)
    return state


def _approval_problems(approval, state, node_id):
    if approval is None:
        return ["gate approval file missing"]
    resp = approval.get("gate_response") or {}
    det = resp.get("details") or {}
    checks = [
        (approval.get("node_id") == node_id and resp.get("node_id", node_id) == node_id, "approval belongs to another node"),
        (approval.get("target_commit_sha") == state["head"], "approval is for a different commit"),
        (approval.get("state_hash") == state["state_hash"], "approval is for a different code/input state (state_hash)"),
        (resp.get("passed") is True, "gate response is not 'passed'"),
        (det.get("stage_c_approved") is True, "Stage C not approved in the gate response"),
        (det.get("commit_match") is True, "audited commit does not match in the gate response"),
        (det.get("clean_workspace") is True, "gate response reports a dirty workspace"),
        (not det.get("unresolved_criticals"), "unresolved critical requirements in the gate response"),
    ]
    return [msg for ok, msg in checks if not ok]


def start_run(workdir, stage, repo=None, inputs=(), preload=(), argv=None, node_id=NODE_ID):
    repo = repo or repo_root()
    state = measure_state(repo, inputs, preload)
    path = os.path.join(workdir, f"gate_approval_{stage}.json")
    approval = json.load(open(path)) if os.path.exists(path) else None
    problems = []
    if state["dirty"]:
        problems.append(f"modified or untracked repository files: {state['dirty']}")
    problems += _approval_problems(approval, state, node_id)
    _manifest(workdir, stage, "start", state, argv if argv is not None else sys.argv,
              {"gate_problems": problems, "gate_verified": not problems, "gate_approval": approval})
    if problems:
        raise SystemExit("refusing to run: " + "; ".join(problems) +
                         f" | HEAD={state['head']} state_hash={state['state_hash']} (run `f1_guard.py preflight`, call the gate, save the approval)")
    return state


def guard_main(workdir, stage, inputs=(), preload=(), argv=None):
    """Entry used by every stage script: `--preflight` prints the state (and exits); otherwise the gate is enforced."""
    argv = list(sys.argv if argv is None else argv)
    if "--preflight" in argv:
        st = preflight(workdir, stage, inputs=inputs, preload=preload, argv=argv)
        print(json.dumps({"stage": stage, "head": st["head"], "dirty": st["dirty"], "state_hash": st["state_hash"],
                          "n_code_files": len(st["code"])}, indent=1))
        raise SystemExit(0)
    return start_run(workdir, stage, inputs=inputs, preload=preload, argv=argv)


def end_run(workdir, stage, start_state, repo=None, inputs=(), argv=None):
    """Re-measure after the run: modules imported lazily during the run must also equal HEAD and the start state."""
    repo = repo or repo_root()
    end = measure_state(repo, inputs)
    new_files = sorted(set(end["code"]) - set(start_state["code"]))
    changed = sorted(f for f, h in end["code"].items() if f in start_state["code"] and start_state["code"][f] != h)
    ok = not end["dirty"] and not changed and end["head"] == start_state["head"]
    _manifest(workdir, stage, "end", end, argv if argv is not None else sys.argv,
              {"modules_loaded_after_start": new_files, "modules_changed_during_run": changed, "run_state_consistent": ok})
    if not ok:
        raise SystemExit(f"run INVALID: HEAD/dirty/modules changed during the run: dirty={end['dirty']} changed={changed}")
    return end


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "preflight":
        st = preflight(sys.argv[2], sys.argv[3])
        print(json.dumps({"head": st["head"], "dirty": st["dirty"], "state_hash": st["state_hash"],
                          "n_code_files": len(st["code"])}, indent=1))
    else:
        raise SystemExit("usage: f1_guard.py preflight <workdir> <stage>")
