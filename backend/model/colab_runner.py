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

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("colab_runner")

_COMPOSE_FILE = Path(__file__).resolve().parent.parent.parent / "infra" / "colab-cli" / "docker-compose.yml"

# Must match docker-compose.yml's colab-cli service: `../../notebooks:/notebooks` (bind-mounted
# from the algoforge/notebooks directory that generate_colab_notebook.py writes into) — same
# default backend/model/colab_trainer.py uses for ALGOFORGE_NOTEBOOKS_DIR, since both need to
# agree on what host directory is visible inside the container at /notebooks.
_NOTEBOOKS_HOST_DIR = Path(os.getenv("ALGOFORGE_NOTEBOOKS_DIR", "../notebooks")).resolve()

# Retry budget for download_with_retry() -- see its docstring. Total worst-case wait is
# sum(delays) + one delay repeated for every attempt beyond len(delays): with the defaults below,
# 6 attempts over 15+30+60+120+180s (the last delay repeats) = up to ~6.75 minutes of patient
# waiting before a one-shot post-exec fetch gives up.
_RETRIEVE_RETRY_ATTEMPTS = 6
_RETRIEVE_RETRY_DELAYS: tuple[float, ...] = (15.0, 30.0, 60.0, 120.0, 180.0)


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

    Single attempt, no retry -- see download_with_retry() below for that. Kept bare here because
    this is also what _poll_and_maybe_stop calls every _POLL_INTERVAL_SECONDS for progress.json;
    that call site already tolerates failure by design (the next poll cycle is its own retry, at
    a cadence tuned against how promptly stop_requested needs to be noticed -- see
    colab_trainer.py's docstrings), so stacking a blocking retry-with-sleep in here as well would
    only add latency to every missed poll for no benefit, and risks measurably slowing down the
    verified-live "stop within one poll cycle" behavior specifically when the connection is
    already degraded, i.e. exactly when a prompt stop matters most.
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

    # Belt-and-suspenders: confirmed live that `colab download` reporting success (returncode 0)
    # does not guarantee the file actually landed where expected (see the local_path docstring
    # note above about a bad path silently swallowing the file inside the container instead).
    if not local_path.is_file():
        raise ColabCliError(
            f"colab download -s {session} {remote_path} reported success but {local_path} was not created"
        )


def download_with_retry(
    session: str, remote_path: str, local_path: Path,
    *, attempts: int = _RETRIEVE_RETRY_ATTEMPTS, delay_schedule: tuple[float, ...] = _RETRIEVE_RETRY_DELAYS,
) -> None:
    """Same as download(), but for a call site where there is no natural next attempt coming on
    its own -- i.e. a one-shot "this must succeed or the run's whole result is lost" fetch, not a
    periodic poll. Retries on ColabCliError with a growing delay between attempts (the schedule
    tuple's last value repeats for any attempt beyond its length).

    Use this at colab_trainer.py's final best.pt/metrics.json retrieval after exec_notebook
    returns, NOT at _poll_and_maybe_stop's per-epoch progress.json poll (that one wants
    download()'s fail-fast behavior -- see download()'s own docstring for why).

    The default schedule (see _RETRIEVE_RETRY_DELAYS) budgets several minutes total, because by
    the time this runs, `colab exec` has already returned -- there is no more time pressure from
    the training itself, so patiently waiting out a slow-to-recover colab-cli connection is pure
    upside against the alternative of discarding a completed run's only copy of its result.
    Observed live on a real ~70-minute run: `colab download` for best.pt/metrics.json started
    failing with "File or directory not found" right as `colab exec` returned, with
    progress.json polls also failing for the ~10 minutes before that -- i.e. a connection outage
    on the order of minutes, not seconds, which is what this schedule is sized against (a fixed
    3x15s budget tried first was not long enough to reliably outlast an outage like that one).
    """
    import time

    last_exc: ColabCliError | None = None
    for attempt in range(1, attempts + 1):
        try:
            download(session, remote_path, local_path)
            return
        except ColabCliError as exc:
            last_exc = exc
            if attempt < attempts:
                delay = delay_schedule[min(attempt, len(delay_schedule)) - 1]
                logger.warning(
                    f"colab_runner.download_with_retry: attempt {attempt}/{attempts} for "
                    f"{remote_path} failed ({exc}); retrying in {delay:.0f}s"
                )
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def stop_session(session: str) -> None:
    """Best-effort — a session left running costs quota but isn't otherwise harmful, so a
    failure here is logged by the caller rather than masking the training result."""
    _run("stop", "-s", session, timeout=60)
