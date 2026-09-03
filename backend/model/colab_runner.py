"""
Thin subprocess wrapper around the `colab-cli` container (see `infra/colab-cli/README.md` for
what each subcommand does and its documented unknowns — exit-code reliability, whether `colab
log` reflects live progress). Assumes the container is already built, running, and
authenticated (`docker compose up -d` + one-time interactive login, both done by a human) —
this module does not attempt any of that itself, and raises ColabCliError with the command's own
stdout/stderr if a step fails rather than guessing at a cause.

Used by model/colab_trainer.py, which is what a TrainingRun with execution_target="colab"
actually runs.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_COMPOSE_FILE = Path(__file__).resolve().parent.parent.parent / "infra" / "colab-cli" / "docker-compose.yml"

# Must match docker-compose.yml's colab-cli service: `../../notebooks:/notebooks` (bind-mounted
# from the algoforge/notebooks directory that generate_colab_notebook.py writes into) — same
# default backend/model/colab_trainer.py uses for ALGOFORGE_NOTEBOOKS_DIR, since both need to
# agree on what host directory is visible inside the container at /notebooks.
_NOTEBOOKS_HOST_DIR = Path(os.getenv("ALGOFORGE_NOTEBOOKS_DIR", "../notebooks")).resolve()


class ColabCliError(RuntimeError):
    """Raised when a `colab` subcommand exits non-zero, or the docker compose invocation itself
    fails (e.g. the container isn't running)."""


def _run(*args: str, timeout: float) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-f", str(_COMPOSE_FILE), "exec", "-T", "colab-cli", "colab", *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ColabCliError(f"colab {' '.join(args)} timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise ColabCliError("docker CLI not found on PATH -- is Docker Desktop installed/running?") from exc
    if result.returncode != 0:
        raise ColabCliError(
            f"colab {' '.join(args)} exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def new_session(session: str) -> None:
    _run("new", "-s", session, timeout=180)


def exec_notebook(session: str, notebook_path_in_container: str, timeout_seconds: float) -> str:
    """Blocks until the notebook finishes or *timeout_seconds* elapses (colab-cli's own
    --timeout). Returns stdout. The subprocess timeout here is deliberately looser than
    --timeout so a slow-to-report CLI failure surfaces as colab-cli's own error, not a generic
    ColabCliError timeout."""
    result = _run(
        "exec", "-s", session, "-f", notebook_path_in_container, "--timeout", str(timeout_seconds),
        timeout=timeout_seconds + 120,
    )
    return result.stdout


def download(session: str, remote_path: str, local_path: Path) -> None:
    """*remote_path* may be given relative to Colab's /content working directory (e.g. the bare
    filename a notebook wrote via a relative path, since that's the notebook's own cwd —
    confirmed live) or as an absolute path; either way this normalizes it, because unlike
    `colab ls` (which defaults its own path argument to "content"), `colab download` does NOT
    resolve a bare relative path against /content itself -- `colab download -s x best.pt out`
    fails with "File or directory not found: best.pt" even when the file is confirmed to exist
    at /content/best.pt, while `.../content/best.pt out` or `.../  /content/best.pt out` both
    succeed.

    *local_path* must resolve under _NOTEBOOKS_HOST_DIR. This is not a style preference --
    `colab`'s own local-path argument is resolved INSIDE the colab-cli container, not on this
    host. Confirmed live: passing a bare Windows host path (e.g.
    C:\\...\\notebooks\\best.pt) makes `colab download` report success and even echo that exact
    path back in its "Downloaded ... to ..." message, while creating nothing at all on the host
    -- it silently wrote a file *named* that whole path string inside the container's own
    filesystem instead. Converting to the container-side /notebooks/... path (which the
    docker-compose.yml bind mount then makes appear at the corresponding host path automatically)
    is the only way this actually lands where the caller expects.
    """
    local_path = local_path.resolve()
    try:
        rel = local_path.relative_to(_NOTEBOOKS_HOST_DIR)
    except ValueError as exc:
        raise ColabCliError(
            f"colab_runner.download: local_path {local_path} must be under {_NOTEBOOKS_HOST_DIR} "
            "(the host directory bind-mounted into the colab-cli container at /notebooks) -- "
            "see this function's docstring for why a path outside it silently fails."
        ) from exc
    container_local_path = f"/notebooks/{rel.as_posix()}"

    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not remote_path.startswith("/"):
        remote_path = f"/content/{remote_path}"
    _run("download", "-s", session, remote_path, container_local_path, timeout=300)

    # Belt-and-suspenders: the above already demonstrated that `colab download` reporting
    # success (returncode 0) does not guarantee the file actually landed where expected.
    if not local_path.is_file():
        raise ColabCliError(
            f"colab download -s {session} {remote_path} reported success but {local_path} was not created"
        )


def stop_session(session: str) -> None:
    """Best-effort — a session left running costs quota but isn't otherwise harmful, so a
    failure here is logged by the caller rather than masking the training result."""
    _run("stop", "-s", session, timeout=60)
