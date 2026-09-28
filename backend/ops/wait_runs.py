"""Wait for AlgoForge training runs through the REST API -- one definition of "done".

Every monitoring failure in the transfer investigation (docs/research-periodic-forcing-transfer.md)
came from ad-hoc waiting loops: watching an orchestrator process instead of the runs, inferring
stalls from log mtimes, backgrounded commands with no timeout. This module fixes the rules once:

  - terminal statuses: completed, error, stopped (and not_found for an unknown id)
  - a stall is `heartbeat_stale == true` from the API (R-13) -- never a log-file heuristic
  - a timeout is REQUIRED; there is no "wait forever"
  - repeated API failures end the wait as "unreachable" instead of spinning silently

It talks only to GET /api/v1/training-runs/status?run_ids=..., so it works from any process that
can reach the API (no DB access, no Celery), stdlib only.

CLI (prints one line per status change, then a JSON summary; exit code = outcome):

    python -m ops.wait_runs 1622 1623 1624 --timeout 7200
    python -m ops.wait_runs 1622 --timeout 600 --interval 30 --api http://localhost:8000

Exit codes: 0 completed, 1 failed (a run ended error/stopped/not_found), 2 stale (a running run
stopped heartbeating), 3 timeout, 4 unreachable, 64 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

TERMINAL = {"completed", "error", "stopped", "not_found"}
EXIT_CODES = {"completed": 0, "failed": 1, "stale": 2, "timeout": 3, "unreachable": 4}
DEFAULT_API = os.getenv("ALGOFORGE_API_URL", "http://localhost:8000")


@dataclass
class WaitResult:
    outcome: str                      # completed | failed | stale | timeout | unreachable
    runs: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.outcome]


def fetch_status(run_ids: list[int], api: str = DEFAULT_API, api_key: str | None = None,
                 timeout: float = 30.0) -> list[dict]:
    url = f"{api.rstrip('/')}/api/v1/training-runs/status?run_ids={','.join(map(str, run_ids))}"
    req = urllib.request.Request(url)
    key = api_key or os.getenv("ALGOFORGE_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["data"]


def wait_for_runs(
    run_ids: list[int],
    timeout: float,
    interval: float = 60.0,
    stop_on_stale: bool = True,
    max_fetch_failures: int = 5,
    fetch: Callable[[list[int]], list[dict]] | None = None,
    on_event: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> WaitResult:
    """Poll until every run is terminal, a run goes stale, the timeout passes, or the API stays
    unreachable for `max_fetch_failures` consecutive polls."""
    if timeout <= 0:
        raise ValueError("timeout must be positive -- waiting forever is not supported")
    fetch = fetch or (lambda ids: fetch_status(ids))
    start = clock()
    last_seen: dict[int, tuple] = {}
    failures = 0
    runs: list[dict] = []
    while True:
        elapsed = clock() - start
        try:
            runs = fetch(run_ids)
            failures = 0
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            failures += 1
            on_event(f"[{elapsed:7.0f}s] status fetch failed ({failures}/{max_fetch_failures}): {e}")
            if failures >= max_fetch_failures:
                return WaitResult("unreachable", runs, elapsed)
        else:
            for r in runs:
                key = (r.get("status"), r.get("heartbeat_stale"), r.get("best_epoch"))
                if last_seen.get(r["run_id"]) != key:
                    last_seen[r["run_id"]] = key
                    extra = f" val_loss={r['val_loss']:.4f}" if r.get("val_loss") is not None else ""
                    stale = " HEARTBEAT STALE" if r.get("heartbeat_stale") else ""
                    err = f" error={r['error_message']}" if r.get("error_message") else ""
                    on_event(f"[{elapsed:7.0f}s] run {r['run_id']}: {r.get('status')}{extra}{stale}{err}")
            if stop_on_stale and any(r.get("heartbeat_stale") for r in runs):
                return WaitResult("stale", runs, elapsed)
            if runs and all(r.get("status") in TERMINAL for r in runs):
                ok = all(r.get("status") == "completed" for r in runs)
                return WaitResult("completed" if ok else "failed", runs, elapsed)
        if elapsed >= timeout:
            return WaitResult("timeout", runs, elapsed)
        sleep(min(interval, max(0.0, timeout - elapsed)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Wait for AlgoForge training runs to finish.")
    p.add_argument("run_ids", type=int, nargs="+")
    p.add_argument("--timeout", type=float, required=True, help="seconds; required")
    p.add_argument("--interval", type=float, default=60.0, help="poll period in seconds (default 60)")
    p.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default {DEFAULT_API})")
    p.add_argument("--no-stop-on-stale", action="store_true",
                   help="keep waiting when a run's heartbeat goes stale (the API reaper will end it)")
    try:
        args = p.parse_args(argv)
    except SystemExit:
        return 64
    res = wait_for_runs(
        args.run_ids, timeout=args.timeout, interval=args.interval,
        stop_on_stale=not args.no_stop_on_stale,
        fetch=lambda ids: fetch_status(ids, api=args.api),
        on_event=lambda line: print(line, flush=True),
    )
    print(json.dumps({"outcome": res.outcome, "elapsed_seconds": round(res.elapsed_seconds),
                      "runs": res.runs}), flush=True)
    return res.exit_code


if __name__ == "__main__":
    sys.exit(main())
