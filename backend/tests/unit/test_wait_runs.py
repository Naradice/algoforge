"""ops.wait_runs: one definition of 'done' for monitoring training runs."""
import urllib.error

import pytest

from ops.wait_runs import wait_for_runs


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _run(rid, status, stale=None, err=None):
    return {"run_id": rid, "status": status, "heartbeat_stale": stale, "val_loss": None,
            "best_epoch": None, "error_message": err}


def _waiter(sequence, **kw):
    clock = FakeClock()
    calls = iter(sequence)

    def fetch(ids):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    events = []
    res = wait_for_runs([1, 2], fetch=fetch, clock=clock, sleep=clock.sleep, on_event=events.append,
                        interval=10, **kw)
    return res, events


def test_completes_when_all_runs_complete():
    res, events = _waiter([
        [_run(1, "running"), _run(2, "pending")],
        [_run(1, "completed"), _run(2, "running")],
        [_run(1, "completed"), _run(2, "completed")],
    ], timeout=100)
    assert res.outcome == "completed" and res.exit_code == 0
    assert sum("run 2: running" in e for e in events) == 1     # only status changes are reported


def test_failed_when_any_run_ends_in_error():
    res, _ = _waiter([[_run(1, "completed"), _run(2, "error", err="worker lost")]], timeout=100)
    assert res.outcome == "failed" and res.exit_code == 1


def test_stale_heartbeat_ends_the_wait():
    res, events = _waiter([[_run(1, "running", stale=False), _run(2, "running", stale=True)]], timeout=100)
    assert res.outcome == "stale" and res.exit_code == 2
    assert any("HEARTBEAT STALE" in e for e in events)


def test_stale_can_be_waited_through():
    res, _ = _waiter([
        [_run(1, "completed"), _run(2, "running", stale=True)],
        [_run(1, "completed"), _run(2, "error", err="worker lost")],
    ], timeout=100, stop_on_stale=False)
    assert res.outcome == "failed"


def test_timeout():
    res, _ = _waiter([[_run(1, "running"), _run(2, "running")]] * 20, timeout=25)
    assert res.outcome == "timeout" and res.exit_code == 3
    assert 25 <= res.elapsed_seconds <= 30


def test_unreachable_after_consecutive_failures():
    err = urllib.error.URLError("refused")
    res, _ = _waiter([err, err, err], timeout=1000, max_fetch_failures=3)
    assert res.outcome == "unreachable" and res.exit_code == 4


def test_a_single_failure_is_tolerated():
    err = urllib.error.URLError("refused")
    res, _ = _waiter([err, [_run(1, "completed"), _run(2, "completed")]], timeout=1000)
    assert res.outcome == "completed"


def test_timeout_is_required():
    with pytest.raises(ValueError):
        wait_for_runs([1], timeout=0, fetch=lambda ids: [])
