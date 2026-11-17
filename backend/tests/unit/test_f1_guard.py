"""F1 (#889) R-15: run guard. Measures the real git/code state, enforces a negative gate decision, and cannot be bypassed by an
environment variable or flag. Uses a throw-away git repository."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

RESEARCH = Path(__file__).resolve().parents[2] / "research"
sys.path.insert(0, str(RESEARCH))

import f1_guard as fg  # noqa: E402

GOOD_RESPONSE = {"node_id": 889, "passed": True, "gate_reason": "ok",
                 "details": {"plan_approved": True, "stage_c_approved": True, "commit_match": True, "clean_workspace": True,
                             "unresolved_criticals": []}}


class Repo:
    def __init__(self, tmp_path, name):
        self.root = tmp_path / "repo"
        (self.root / "backend" / "research").mkdir(parents=True)
        self.mod_name = name
        self.mod = self.root / "backend" / "research" / f"{name}.py"
        self.mod.write_text("VALUE = 1\n")
        self.git("init", "-q")
        self.git("config", "user.email", "t@t")
        self.git("config", "user.name", "t")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "init")
        sys.path.insert(0, str(self.root / "backend" / "research"))
        importlib.invalidate_caches()
        importlib.import_module(name)

    def git(self, *a):
        return subprocess.run(["git", "-C", str(self.root), *a], check=True, capture_output=True, text=True).stdout.strip()

    def head(self):
        return self.git("rev-parse", "HEAD")

    def close(self):
        sys.path.remove(str(self.root / "backend" / "research"))
        sys.modules.pop(self.mod_name, None)

    def approval(self, workdir, stage="s", **override):
        st = fg.measure_state(self.root)
        a = {"node_id": 889, "target_commit_sha": st["head"], "state_hash": st["state_hash"],
             "gate_response": json.loads(json.dumps(GOOD_RESPONSE))}
        for k, v in override.items():
            if k.startswith("det_"):
                a["gate_response"]["details"][k[4:]] = v
            elif k == "passed":
                a["gate_response"]["passed"] = v
            else:
                a[k] = v
        Path(workdir).mkdir(parents=True, exist_ok=True)
        (Path(workdir) / f"gate_approval_{stage}.json").write_text(json.dumps(a))
        return a


@pytest.fixture()
def repo(tmp_path):
    r = Repo(tmp_path, "f1_dep_guard")
    yield r
    r.close()


def run(repo, tmp_path, stage="s"):
    return fg.start_run(str(tmp_path / "w"), stage, repo=str(repo.root), argv=["x"])


# R-15
def test_state_hash_is_stable_and_changes_with_the_code(repo):
    s1 = fg.measure_state(repo.root)
    assert s1["head"] == repo.head() and s1["dirty"] == [] and "backend/research/f1_dep_guard.py" in s1["code"]
    assert fg.measure_state(repo.root)["state_hash"] == s1["state_hash"]
    repo.mod.write_text("VALUE = 2\n")
    s2 = fg.measure_state(repo.root)
    assert s2["state_hash"] != s1["state_hash"] and s2["dirty"] == ["backend/research/f1_dep_guard.py"]


# R-15
def test_refuses_without_an_approval_but_records_the_attempt_and_the_state(repo, tmp_path):
    with pytest.raises(SystemExit) as e:
        run(repo, tmp_path)
    assert "approval file missing" in str(e.value) and repo.head() in str(e.value)
    manifests = list((tmp_path / "w").glob("run_manifest_s_start_*.json"))
    assert manifests and json.load(open(manifests[0]))["gate_verified"] is False


# R-15
def test_accepts_a_passing_approval_for_exactly_this_state(repo, tmp_path):
    repo.approval(tmp_path / "w")
    st = run(repo, tmp_path)
    m = json.load(open(next((tmp_path / "w").glob("run_manifest_s_start_*.json"))))
    assert st["head"] == repo.head() and m["gate_verified"] is True and m["gate_problems"] == []
    end = fg.end_run(str(tmp_path / "w"), "s", st, repo=str(repo.root), argv=["x"])
    assert end["state_hash"] == st["state_hash"]


@pytest.mark.parametrize("override,text", [
    ({"target_commit_sha": "0" * 40}, "different commit"),
    ({"state_hash": "f" * 64}, "different code/input state"),
    ({"node_id": 1}, "another node"),
    ({"passed": False}, "not 'passed'"),
    ({"det_stage_c_approved": False}, "Stage C not approved"),
    ({"det_commit_match": False}, "does not match"),
    ({"det_clean_workspace": False}, "dirty workspace"),
    ({"det_unresolved_criticals": ["R-10b"]}, "unresolved critical"),
])
# R-15
def test_refuses_every_kind_of_wrong_approval(repo, tmp_path, override, text):
    repo.approval(tmp_path / "w", **override)
    with pytest.raises(SystemExit) as e:
        run(repo, tmp_path)
    assert text in str(e.value)


# R-15
def test_environment_variables_cannot_bypass_the_guard(repo, tmp_path, monkeypatch):
    for k, v in (("F1_ALLOW_DIRTY", "1"), ("F1_EXPECTED_COMMIT", repo.head()), ("F1_SKIP_GUARD", "1")):
        monkeypatch.setenv(k, v)
    with pytest.raises(SystemExit):
        run(repo, tmp_path)                                     # no approval: refused despite the variables
    repo.mod.write_text("VALUE = 3\n")                          # dirty code with an approval issued for the dirty state
    repo.approval(tmp_path / "w")
    with pytest.raises(SystemExit) as e:
        run(repo, tmp_path)
    assert "modified or untracked" in str(e.value)


# R-15
def test_an_untracked_module_that_is_loaded_is_refused(repo, tmp_path):
    extra = repo.root / "backend" / "research" / "f1_dep_guard_extra.py"
    extra.write_text("Y = 1\n")
    importlib.invalidate_caches()
    importlib.import_module("f1_dep_guard_extra")
    try:
        repo.approval(tmp_path / "w")
        with pytest.raises(SystemExit) as e:
            run(repo, tmp_path)
        assert "f1_dep_guard_extra.py" in str(e.value)
    finally:
        sys.modules.pop("f1_dep_guard_extra", None)


# R-15
def test_preload_brings_lazily_imported_modules_into_the_measured_state(repo):
    lazy = repo.root / "backend" / "research" / "f1_dep_lazy.py"
    lazy.write_text("Z = 1\n")
    repo.git("add", ".")
    repo.git("commit", "-q", "-m", "lazy")
    importlib.invalidate_caches()
    try:
        before = fg.measure_state(repo.root)
        after = fg.measure_state(repo.root, preload=["f1_dep_lazy"])
        assert "backend/research/f1_dep_lazy.py" not in before["code"] and "backend/research/f1_dep_lazy.py" in after["code"]
        assert before["state_hash"] != after["state_hash"]
    finally:
        sys.modules.pop("f1_dep_lazy", None)


# R-15
def test_end_run_flags_a_modification_made_during_the_run(repo, tmp_path):
    repo.approval(tmp_path / "w")
    st = run(repo, tmp_path)
    repo.mod.write_text("VALUE = 99\n")
    with pytest.raises(SystemExit) as e:
        fg.end_run(str(tmp_path / "w"), "s", st, repo=str(repo.root), argv=["x"])
    assert "INVALID" in str(e.value)
    end_manifest = json.load(open(next((tmp_path / "w").glob("run_manifest_s_end_*.json"))))
    assert end_manifest["run_state_consistent"] is False and end_manifest["modules_changed_during_run"]


# R-15
def test_stage_scripts_call_the_guard_before_doing_anything():
    """Every executable stage script must route through guard_main and end_run (no path around the guard)."""
    for name, stages in (("f1_ddm.py", 1), ("f1_run.py", 1), ("f1_analyze.py", 1)):
        src = (RESEARCH / name).read_text()
        main = src[src.index('if __name__ == "__main__":'):]
        assert "fg.guard_main(" in main and "fg.end_run(" in main, name
        assert main.index("fg.guard_main(") < min(main.index(x) for x in ("stage_map(", "stage_score(", "stage_select")
                                                    if x in main), name
        assert "os.environ" not in main and "getenv" not in main, name
