
"""Regression tests — the startup-time capability-load failure registry.

app.py loads routers/modules under a broad try/except; a failure was only
logged, so the server still started and /health reported healthy while the
affected endpoint silently 404'd (capabilities dropped silently, hard to
diagnose). startup_state records load failures so /health/detailed can report
which capabilities failed to load.

Pure logic, no external dependencies.
"""

import pytest

from src.api import startup_state


@pytest.fixture(autouse=True)
def _clean():
    startup_state.clear_load_failures()
    yield
    startup_state.clear_load_failures()


def test_record_and_get():
    startup_state.record_load_failure("rag_indexing", ImportError("boom"))
    failures = startup_state.get_load_failures()
    assert "rag_indexing" in failures
    # Store the exception type name (do not leak the full message; details go to the log)
    assert failures["rag_indexing"] == "ImportError"


def test_get_returns_copy_not_internal():
    """The return value should be a copy; the caller mutating it must not affect internal state."""
    startup_state.record_load_failure("m", ValueError("x"))
    got = startup_state.get_load_failures()
    got["injected"] = "y"
    assert "injected" not in startup_state.get_load_failures()


def test_clear():
    startup_state.record_load_failure("m", RuntimeError("x"))
    startup_state.clear_load_failures()
    assert startup_state.get_load_failures() == {}


def test_string_error_accepted():
    """A plain string reason (not an exception object) is also accepted."""
    startup_state.record_load_failure("m", "custom reason")
    assert startup_state.get_load_failures()["m"] == "custom reason"
