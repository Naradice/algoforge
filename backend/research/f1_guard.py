"""F1 (#889): run guard. Every executable stage calls `start_run` first.

* records the executed commit, whether the F1 files are dirty, the command line, package versions and input-file hashes in
  `run_manifest_<stage>_<utc time>.json` inside the work directory (post-run audit: executed commit == audited commit);
* refuses to run if `F1_EXPECTED_COMMIT` is set and differs from HEAD, or if the F1 files are modified/untracked
  (unless `F1_ALLOW_DIRTY=1`, which is recorded in the manifest).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

F1_PATHS = ["backend/research", "backend/tests/unit", "backend/requirements.txt"]
F1_PATTERNS = ("f1_", "test_f1", "requirements.txt")


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def repo_root():
    return _git(os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--show-toplevel").strip()


def git_state(repo):
    head = _git(repo, "rev-parse", "HEAD").strip()
    status = _git(repo, "status", "--porcelain", "--", *F1_PATHS).splitlines()
    dirty = [ln for ln in status if any(p in os.path.basename(ln[3:].strip()) for p in F1_PATTERNS)]
    return head, dirty


def sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def start_run(workdir, stage, repo=None, inputs=None, argv=None):
    repo = repo or repo_root()
    head, dirty = git_state(repo)
    expected = os.environ.get("F1_EXPECTED_COMMIT")
    allow_dirty = os.environ.get("F1_ALLOW_DIRTY") == "1"
    manifest = {
        "stage": stage, "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "head": head, "expected_commit": expected,
        "commit_matches_expected": (expected == head) if expected else None, "dirty_files": dirty,
        "allow_dirty": allow_dirty, "argv": list(argv if argv is not None else sys.argv), "python": sys.version.split()[0],
        "inputs": {p: sha256(p) for p in (inputs or [])},
    }
    for mod in ("numpy", "scipy", "pandas", "arch", "statsmodels"):
        try:
            manifest[mod] = __import__(mod).__version__
        except Exception:
            manifest[mod] = None
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, f"run_manifest_{stage}_{manifest['utc'][:19].replace(':', '')}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    if expected and expected != head:
        raise SystemExit(f"refusing to run: HEAD {head} != F1_EXPECTED_COMMIT {expected}")
    if dirty and not allow_dirty:
        raise SystemExit(f"refusing to run: F1 files are modified or untracked: {dirty}")
    return manifest
