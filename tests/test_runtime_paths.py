
"""Unit tests for src/utils/runtime_paths.py

No dependency on the DB / vLLM / any external service — uses tmp_path +
monkeypatch to simulate the env and sys state of the dev / Nuitka onefile /
standalone deployment scenarios. Directory names always carry a uuid to avoid
accidentally matching a real python bin directory or a same-named directory
under /app.
"""

import shutil
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from src.utils.runtime_paths import resolve_base_dir, resolve_external_dir

_LINUX = sys.platform.startswith("linux")


def _unique_name() -> str:
    """Generate a resource directory name that cannot collide."""
    return f"res_{uuid4().hex}"


@pytest.fixture
def no_nuitka_env(monkeypatch):
    """Ensure the test starts with no Nuitka onefile env signal."""
    monkeypatch.delenv("NUITKA_ONEFILE_PARENT", raising=False)


def test_external_dir_found_under_dev_root(tmp_path, no_nuitka_env):
    """dev mode: when dev_root/name exists, that directory is returned (TC-runtime-paths-01)."""
    name = _unique_name()
    (tmp_path / name).mkdir()
    assert resolve_external_dir(name, dev_root=tmp_path) == tmp_path / name


def test_external_dir_returns_none_when_nowhere(tmp_path, no_nuitka_env):
    """When no candidate location exists, None is returned (TC-runtime-paths-02)."""
    assert resolve_external_dir(_unique_name(), dev_root=tmp_path) is None


def test_external_dir_ignores_plain_file(tmp_path, no_nuitka_env):
    """A same-named plain file (not a directory) is not a hit; None is returned (TC-runtime-paths-03)."""
    name = _unique_name()
    (tmp_path / name).write_text("not a dir")
    assert resolve_external_dir(name, dev_root=tmp_path) is None


def test_external_dir_sys_executable_beats_dev_root(tmp_path, monkeypatch, no_nuitka_env):
    """A directory beside sys.executable (standalone deployment) takes precedence over dev_root (TC-runtime-paths-04)."""
    name = _unique_name()
    bin_dir = tmp_path / "bin"
    (bin_dir / name).mkdir(parents=True)
    dev_root = tmp_path / "dev"
    (dev_root / name).mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(bin_dir / "app.bin"))
    assert resolve_external_dir(name, dev_root=dev_root) == bin_dir / name


def test_external_dir_rejects_tmp_onefile_executable(tmp_path, monkeypatch, no_nuitka_env):
    """When sys.executable is under /tmp/onefile_*, it is rejected and falls back to dev_root (TC-runtime-paths-05)."""
    name = _unique_name()
    onefile_dir = Path(tempfile.mkdtemp(prefix="onefile_", dir="/tmp"))
    try:
        (onefile_dir / name).mkdir()  # even a bundled copy in the extraction dir must be rejected
        monkeypatch.setattr(sys, "executable", str(onefile_dir / "app.bin"))
        dev_root = tmp_path / "dev"
        (dev_root / name).mkdir(parents=True)
        assert resolve_external_dir(name, dev_root=dev_root) == dev_root / name
    finally:
        shutil.rmtree(onefile_dir, ignore_errors=True)


def test_base_dir_dev_mode_returns_dev_root(tmp_path, no_nuitka_env):
    """Non-compiled (plain Python) mode returns dev_root directly (TC-runtime-paths-06)."""
    assert resolve_base_dir(dev_root=tmp_path) == tmp_path


@pytest.mark.skipif(not _LINUX, reason="依賴 /proc/self/exe(Linux only)")
def test_base_dir_frozen_uses_real_executable_dir(tmp_path, monkeypatch, no_nuitka_env):
    """With sys.frozen (compiled), returns the directory of /proc/self/exe, not dev_root (TC-runtime-paths-07)."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    expected = Path("/proc/self/exe").resolve().parent
    assert resolve_base_dir(dev_root=tmp_path) == expected
    assert resolve_base_dir(dev_root=tmp_path) != tmp_path


@pytest.mark.skipif(not _LINUX, reason="依賴 /proc(Linux only)")
def test_base_dir_nuitka_parent_pid_resolves_parent_exe(tmp_path, monkeypatch):
    """When NUITKA_ONEFILE_PARENT points at a live process, returns its executable's directory (TC-runtime-paths-08)."""
    monkeypatch.setenv("NUITKA_ONEFILE_PARENT", "self")  # /proc/self/exe must exist
    expected = Path("/proc/self/exe").resolve().parent
    assert resolve_base_dir(dev_root=tmp_path) == expected


@pytest.mark.skipif(not _LINUX, reason="依賴 /proc/self/exe(Linux only)")
def test_base_dir_invalid_nuitka_pid_falls_back_to_self_exe(tmp_path, monkeypatch):
    """When NUITKA_ONEFILE_PARENT is a dead PID, it is still treated as compiled and falls back to /proc/self/exe (TC-runtime-paths-09)."""
    monkeypatch.setenv("NUITKA_ONEFILE_PARENT", "999999999")
    expected = Path("/proc/self/exe").resolve().parent
    assert resolve_base_dir(dev_root=tmp_path) == expected
    assert resolve_base_dir(dev_root=tmp_path) != tmp_path


@pytest.mark.skipif(not _LINUX, reason="依賴 /proc/self/exe(Linux only)")
def test_base_dir_nuitka_compiled_marker_detected(tmp_path, monkeypatch, no_nuitka_env):
    """__main__.__compiled__ (the Nuitka marker) also triggers the compiled path (TC-runtime-paths-10)."""
    main_mod = sys.modules["__main__"]
    monkeypatch.setattr(main_mod, "__compiled__", True, raising=False)
    expected = Path("/proc/self/exe").resolve().parent
    assert resolve_base_dir(dev_root=tmp_path) == expected


@pytest.mark.skipif(not _LINUX, reason="依賴 /proc(Linux only)")
def test_external_dir_nuitka_parent_candidate_walked(tmp_path, monkeypatch):
    """NUITKA_ONEFILE_PARENT=self: candidate chain (1) is walked (still falls back correctly to dev_root when there is no resource beside the parent exe) (TC-runtime-paths-11)."""
    name = _unique_name()
    monkeypatch.setenv("NUITKA_ONEFILE_PARENT", "self")  # /proc/self/exe must exist
    (tmp_path / name).mkdir()
    assert resolve_external_dir(name, dev_root=tmp_path) == tmp_path / name


def test_external_dir_dead_nuitka_pid_skipped(tmp_path, monkeypatch):
    """NUITKA_ONEFILE_PARENT is a dead PID: candidate (1) is silently skipped without crashing, fallback works (TC-runtime-paths-12)."""
    name = _unique_name()
    monkeypatch.setenv("NUITKA_ONEFILE_PARENT", "999999999")
    (tmp_path / name).mkdir()
    assert resolve_external_dir(name, dev_root=tmp_path) == tmp_path / name


def test_base_dir_non_linux_compiled_uses_sys_executable(tmp_path, monkeypatch, no_nuitka_env):
    """compiled + non-Linux (candidate (2) /proc unavailable) → candidate (3) the sys.executable directory (TC-runtime-paths-13)."""
    import src.utils.runtime_paths as rp
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(rp.sys, "platform", "win32")
    expected = Path(sys.executable).resolve().parent
    assert resolve_base_dir(dev_root=tmp_path) == expected
