"""Code + environment fingerprint for a training run, so a finished run can be re-run later.

TrainingRun already records *what* was trained (model config, hyperparams, dataset ids,
data_provenance) but not *with which code and packages*: two runs with identical hyperparams can
differ because model_core changed in between, or because torch was upgraded. capture() is
called once per run, when the worker flips it to running (celery_worker._resolve_training_context),
and the result is stored in TrainingRun.run_environment.

Shape (every key always present; None when unknown):
    git_commit              full SHA of HEAD in the algoforge repo
    git_dirty               True if tracked files under backend/ differ from HEAD (untracked
                            files ignored; same scope as ops/worker_registry.code_revision)
    git_dirty_files         up to 50 modified tracked paths
    git_diff_sha256         sha256 of `git diff HEAD -- backend` -- identifies the exact
                            uncommitted state
    git_remote              remote URL with any credentials stripped
    git_unavailable_reason  why git info is missing (e.g. Docker container without .git)
    worker_git_commit       HEAD when this worker process started -- a worker keeps running the
                            code it imported at startup, so if this differs from git_commit the
                            run executed older code than git_commit says; see code_changed_since_worker_start
    code_changed_since_worker_start
    python, platform, torch, cuda, gpu
    packages                sorted "name==version" of the worker's interpreter
    packages_sha256         sha256 of that list, for quick equality checks across runs
    captured_at

Git info comes from `git` in the repo root. Inside the Docker dev containers only backend/ is
mounted (no .git); there, set ALGOFORGE_CODE_REVISION (the same override ops/worker_registry
uses) and optionally ALGOFORGE_CODE_DIRTY=1, or mount the repo's .git. Otherwise git_commit is
None and git_unavailable_reason says why.

Never raises: a fingerprint failure must not fail a training run.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

logger = logging.getLogger(__name__)

# backend/model/run_environment.py -> backend/model -> backend -> algoforge/
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_GIT_TIMEOUT_S = 10
_MAX_DIRTY_FILES = 50
_CREDENTIALS_IN_URL = re.compile(r"(://)[^/@\s]+@")


def _git(*args: str, repo_root: Path) -> str:
    # safe.directory=*: in a container the repo is often owned by another uid, and git then
    # refuses every command ("dubious ownership") -- read-only queries here are harmless.
    return subprocess.check_output(
        ["git", "-c", "safe.directory=*", *args],
        cwd=repo_root, text=True, stderr=subprocess.DEVNULL, timeout=_GIT_TIMEOUT_S,
    )


def git_state(repo_root: Path = REPO_ROOT) -> dict:
    state: dict = {
        "git_commit": None, "git_dirty": None, "git_dirty_files": None, "git_diff_sha256": None,
        "git_remote": None, "git_unavailable_reason": None,
    }
    env_commit = os.getenv("ALGOFORGE_CODE_REVISION")
    try:
        state["git_commit"] = _git("rev-parse", "HEAD", repo_root=repo_root).strip()
    except FileNotFoundError:
        state["git_unavailable_reason"] = "git executable not found"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        state["git_unavailable_reason"] = f"not a git checkout or git failed ({type(exc).__name__})"

    if state["git_commit"] is None:
        if env_commit:
            state["git_commit"] = env_commit.strip()
            dirty = os.getenv("ALGOFORGE_CODE_DIRTY")
            state["git_dirty"] = None if dirty is None else dirty.strip().lower() in ("1", "true", "yes")
            state["git_unavailable_reason"] = "git unavailable; commit taken from ALGOFORGE_CODE_REVISION"
        return state

    try:
        # Tracked changes only: an untracked results/*.json from an analysis script doesn't
        # change what the training code does.
        porcelain = _git("status", "--porcelain", "--untracked-files=no", "--", "backend", repo_root=repo_root)
        dirty_files = [line[3:] for line in porcelain.splitlines() if line.strip()]
        state["git_dirty"] = bool(dirty_files)
        state["git_dirty_files"] = dirty_files[:_MAX_DIRTY_FILES]
        if dirty_files:
            diff = _git("diff", "HEAD", "--", "backend", repo_root=repo_root)
            state["git_diff_sha256"] = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        logger.warning("run_environment: git status failed", exc_info=True)

    try:
        remotes = _git("remote", repo_root=repo_root).split()
        if remotes:
            name = "origin" if "origin" in remotes else remotes[0]
            url = _git("remote", "get-url", name, repo_root=repo_root).strip()
            state["git_remote"] = _CREDENTIALS_IN_URL.sub(r"\1", url)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        pass
    return state


def _packages() -> list[str]:
    seen: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            seen.setdefault(name.lower(), f"{name}=={dist.version}")
    return sorted(seen.values(), key=str.lower)


def _torch_info() -> dict:
    info = {"torch": None, "cuda": None, "gpu": None}
    try:
        import torch
    except Exception:
        return info
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    try:
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:
        pass
    return info


# HEAD at worker-process start. celery_worker imports this module at startup, so this is the
# commit of the code the worker process actually loaded (modules imported later, lazily, can
# still be newer -- see code_changed_since_worker_start).
_WORKER_GIT_STATE: dict | None = None


def record_worker_start() -> None:
    global _WORKER_GIT_STATE
    try:
        _WORKER_GIT_STATE = git_state()
    except Exception:
        logger.warning("run_environment: could not record worker start state", exc_info=True)


def capture() -> dict:
    """Fingerprint for one training run. Never raises; returns {"error": ...} at worst."""
    try:
        state = git_state()
        worker = _WORKER_GIT_STATE or {}
        worker_commit = worker.get("git_commit")
        packages = _packages()
        return {
            **state,
            "worker_git_commit": worker_commit,
            "code_changed_since_worker_start": (
                None if worker_commit is None or state["git_commit"] is None
                else (worker_commit != state["git_commit"] or worker.get("git_diff_sha256") != state["git_diff_sha256"])
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            **_torch_info(),
            "packages": packages,
            "packages_sha256": hashlib.sha256("\n".join(packages).encode("utf-8")).hexdigest(),
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:  # pragma: no cover - defensive; see module docstring
        logger.warning("run_environment: capture failed", exc_info=True)
        return {"error": f"{type(exc).__name__}: {exc}", "captured_at": datetime.now(timezone.utc).isoformat()}
