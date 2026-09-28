"""ops.worker_registry (R-5): worker registry + queue status."""
import fnmatch
import json

import pytest

import ops.worker_registry as wr


class FakeRedis:
    def __init__(self):
        self.kv, self.lists, self.ttl = {}, {}, {}

    def set(self, k, v, ex=None):
        self.kv[k] = v
        self.ttl[k] = ex

    def get(self, k):
        return self.kv.get(k)

    def delete(self, *keys):
        for k in keys:
            self.kv.pop(k, None)

    def scan_iter(self, match="*"):
        return [k for k in list(self.kv) if fnmatch.fnmatch(k, match)]

    def llen(self, k):
        return len(self.lists.get(k, []))


@pytest.fixture
def fake(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(wr, "_redis", lambda: r)
    monkeypatch.setattr(wr, "code_revision", lambda: ("abc123", False))
    return r


def test_registration_publishes_with_ttl_and_tracks_current_task(fake):
    reg = wr.WorkerRegistration("w1@host", ["training"])
    reg._publish()
    info = json.loads(fake.get("algoforge:worker:w1@host"))
    assert info["code_revision"] == "abc123" and info["queues"] == ["training"]
    assert fake.ttl["algoforge:worker:w1@host"] == wr.WORKER_TTL_SECONDS
    reg.set_current_task({"task": "celery_worker.train_model", "args": [1622]})
    assert json.loads(fake.get(reg.key))["current_task"]["args"] == [1622]
    reg.stop()
    assert fake.get(reg.key) is None


def test_queue_status_flags_stale_code_and_counts(fake):
    fake.set("algoforge:worker:old@h", json.dumps({"hostname": "old@h", "code_revision": "zzz999",
                                                   "queues": ["training"], "current_task": {"task": "x"}}))
    fake.set("algoforge:worker:new@h", json.dumps({"hostname": "new@h", "code_revision": "abc123",
                                                   "queues": ["training", "collection"], "current_task": None}))
    fake.lists["training"] = ["m1", "m2"]
    st = wr.get_queue_status()
    by = {w["hostname"]: w for w in st["workers"]}
    assert by["old@h"]["stale_code"] is True and by["new@h"]["stale_code"] is False
    assert st["queues"]["training"] == {"pending": 2, "workers": 2, "busy": 1}
    assert st["queues"]["collection"] == {"pending": 0, "workers": 1, "busy": 0}
    assert st["queues"]["backtest"]["workers"] == 0          # nothing consuming -> caller should not dispatch


def test_unknown_revision_is_never_flagged_stale(fake, monkeypatch):
    monkeypatch.setattr(wr, "code_revision", lambda: ("unknown", False))
    fake.set("algoforge:worker:a@h", json.dumps({"hostname": "a@h", "code_revision": "abc", "queues": []}))
    assert wr.get_queue_status()["workers"][0]["stale_code"] is False
