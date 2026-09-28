"""Worker registry and queue status (requirements.md R-5).

Why not just Celery's inspect(): local workers run with --pool=solo, where the task executes on the
worker's main thread, so a worker busy training cannot answer control commands -- inspect() would
report exactly the busy workers as missing. Instead each worker registers itself in Redis from a
daemon thread (independent of whatever the task is doing):

    algoforge:worker:<hostname>  -> JSON {hostname, pid, started_at, code_revision, code_dirty,
                                         queues, current_task, last_seen, host_memory}
                                    TTL WORKER_TTL_SECONDS, refreshed every REFRESH_SECONDS

A worker that dies stops refreshing and its key expires, so "registered" == "alive within the
TTL". get_queue_status() combines the registry with each queue's pending-message count (the Redis
list the kombu transport keeps per queue) and flags workers whose code revision differs from the
repository's current HEAD -- the stale-worker failure R-5 describes (a worker started before a code
change fails with unrelated-looking errors).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

KEY_PREFIX = "algoforge:worker:"
REFRESH_SECONDS = float(os.getenv("ALGOFORGE_WORKER_REFRESH_SECONDS", "30"))
WORKER_TTL_SECONDS = int(os.getenv("ALGOFORGE_WORKER_TTL_SECONDS", "90"))
QUEUES = ("collection", "characteristics", "training", "backtest", "colab")
_REPO_DIR = Path(__file__).resolve().parents[2]


def _redis():
    import redis as _redis
    return _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)


def code_revision() -> tuple[str, bool]:
    """(git HEAD, whether backend/ has uncommitted changes). ALGOFORGE_CODE_REVISION overrides
    (e.g. a Docker image built without .git)."""
    env = os.getenv("ALGOFORGE_CODE_REVISION")
    if env:
        return env, False
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_DIR, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "backend"], cwd=_REPO_DIR,
                               capture_output=True, timeout=10).returncode != 0
        return head, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


def _host_memory() -> dict | None:
    try:
        import psutil
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        return {"available_gb": round(vm.available / 2**30, 2), "total_gb": round(vm.total / 2**30, 2),
                "swap_free_gb": round(sw.free / 2**30, 2)}
    except Exception:  # noqa: BLE001
        return None


class WorkerRegistration:
    """One per worker process: started from worker_ready, stopped from worker_shutdown."""

    def __init__(self, hostname: str, queues: list[str]):
        rev, dirty = code_revision()
        self.info = {
            "hostname": hostname, "pid": os.getpid(), "machine": socket.gethostname(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "code_revision": rev, "code_dirty": dirty, "queues": sorted(queues), "current_task": None,
        }
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="worker-registry", daemon=True)

    @property
    def key(self) -> str:
        return KEY_PREFIX + self.info["hostname"]

    def start(self) -> "WorkerRegistration":
        self._thread.start()
        return self

    def set_current_task(self, task: dict | None) -> None:
        with self._lock:
            self.info["current_task"] = task
        self._publish()

    def _publish(self) -> None:
        with self._lock:
            payload = {**self.info, "last_seen": datetime.now(timezone.utc).isoformat(),
                       "host_memory": _host_memory()}
        try:
            _redis().set(self.key, json.dumps(payload), ex=WORKER_TTL_SECONDS)
        except Exception as e:  # noqa: BLE001 -- registry is best-effort, never break the worker
            logger.warning(f"worker registry publish failed: {e}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._publish()
            self._stop.wait(REFRESH_SECONDS)

    def stop(self) -> None:
        self._stop.set()
        try:
            _redis().delete(self.key)
        except Exception:  # noqa: BLE001
            pass


def get_queue_status(redis_client=None) -> dict:
    """Queues (pending message count + registered workers consuming them) and workers (with
    stale_code = code revision differs from the repository's current HEAD)."""
    r = redis_client or _redis()
    head, _ = code_revision()
    workers = []
    for key in r.scan_iter(match=KEY_PREFIX + "*"):
        raw = r.get(key)
        if not raw:
            continue
        w = json.loads(raw)
        w["stale_code"] = (head != "unknown" and w.get("code_revision") not in (None, "unknown", head))
        workers.append(w)
    workers.sort(key=lambda w: w["hostname"])
    queues = {}
    for q in QUEUES:
        consumers = [w for w in workers if q in w.get("queues", [])]
        queues[q] = {
            "pending": int(r.llen(q)),
            "workers": len(consumers),
            "busy": sum(1 for w in consumers if w.get("current_task")),
        }
    return {"repo_revision": head, "queues": queues, "workers": workers}
