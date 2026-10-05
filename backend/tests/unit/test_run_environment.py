"""model/run_environment.py: git + package fingerprint recorded on every training run."""
from __future__ import annotations

import subprocess

import pytest

from model import run_environment as re_mod


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv("ALGOFORGE_CODE_REVISION", raising=False)
    monkeypatch.delenv("ALGOFORGE_CODE_DIRTY", raising=False)
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "train.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("readme\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@example.com", "-c", "user.name=t", "add", ".")
    _git(tmp_path, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "init")
    return tmp_path


def _head(repo) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def test_clean_checkout(repo):
    state = re_mod.git_state(repo)
    assert state["git_commit"] == _head(repo)
    assert len(state["git_commit"]) == 40
    assert state["git_dirty"] is False
    assert state["git_dirty_files"] == []
    assert state["git_diff_sha256"] is None
    assert state["git_unavailable_reason"] is None


def test_uncommitted_backend_change_is_dirty_with_a_diff_hash(repo):
    (repo / "backend" / "train.py").write_text("x = 2\n")
    state = re_mod.git_state(repo)
    assert state["git_dirty"] is True
    assert state["git_dirty_files"] == ["backend/train.py"]
    first = state["git_diff_sha256"]
    assert first and len(first) == 64

    (repo / "backend" / "train.py").write_text("x = 3\n")
    assert re_mod.git_state(repo)["git_diff_sha256"] != first  # a different edit, a different hash


def test_untracked_files_and_changes_outside_backend_are_not_dirty(repo):
    (repo / "backend" / "results.json").write_text("{}")  # untracked
    (repo / "README.md").write_text("edited\n")          # tracked, outside backend/
    assert re_mod.git_state(repo)["git_dirty"] is False


def test_remote_credentials_are_stripped(repo):
    _git(repo, "remote", "add", "remote", "https://user:" + "s3cret" + "@github.com/o/algoforge.git")
    state = re_mod.git_state(repo)
    assert state["git_remote"] == "https://github.com/o/algoforge.git"


def test_no_git_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("ALGOFORGE_CODE_REVISION", raising=False)
    state = re_mod.git_state(tmp_path)
    assert state["git_commit"] is None
    assert state["git_dirty"] is None
    assert "not a git checkout" in state["git_unavailable_reason"]


def test_env_override_when_git_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGOFORGE_CODE_REVISION", "abc1234")
    monkeypatch.setenv("ALGOFORGE_CODE_DIRTY", "1")
    state = re_mod.git_state(tmp_path)
    assert state["git_commit"] == "abc1234"
    assert state["git_dirty"] is True
    assert "ALGOFORGE_CODE_REVISION" in state["git_unavailable_reason"]


def test_capture_shape_and_worker_start_comparison(repo, monkeypatch):
    monkeypatch.setattr(re_mod, "REPO_ROOT", repo)
    real_git_state = re_mod.git_state
    monkeypatch.setattr(re_mod, "git_state", lambda repo_root=None: real_git_state(repo))

    re_mod.record_worker_start()
    same = re_mod.capture()
    assert same["git_commit"] == _head(repo)
    assert same["worker_git_commit"] == _head(repo)
    assert same["code_changed_since_worker_start"] is False
    assert same["python"] and same["platform"] and same["captured_at"]
    assert same["torch"]  # torch is a backend dependency
    assert any(p.lower().startswith("sqlalchemy==") for p in same["packages"])
    assert len(same["packages_sha256"]) == 64

    # Code edited after the worker started: the run executes the worker's (older) code.
    (repo / "backend" / "train.py").write_text("x = 99\n")
    changed = re_mod.capture()
    assert changed["code_changed_since_worker_start"] is True


def test_capture_never_raises(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("git exploded")
    monkeypatch.setattr(re_mod, "git_state", boom)
    result = re_mod.capture()
    assert "git exploded" in result["error"]


def test_mcp_view_drops_package_list_unless_asked():
    from mcp_server.tools.model import _run_environment_view

    env = {"git_commit": "a" * 40, "packages": ["a==1", "b==2"], "packages_sha256": "x"}
    assert _run_environment_view(env, include_packages=False) == {"git_commit": "a" * 40, "packages_sha256": "x", "packages_count": 2}
    assert _run_environment_view(env, include_packages=True) == env
    assert _run_environment_view(None, include_packages=False) is None
