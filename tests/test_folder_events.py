
"""Folder event hooks — dependency inversion for MCP tool registration.

FolderAdapter used to import register/unregister directly from
fastmcp_tools.agentic_tools, forming an import cycle with
agentic_tools → adapter.rag. Now folder.py only calls this module's notify_*,
and app.py wires in the MCP layer's register/unregister at startup, so the
dependency direction is one-way (fastmcp → adapter).

Contract pinned by these tests:
1. Once wired, created/renamed/deleted dispatch to the corresponding callback
   with the arguments unchanged.
2. When unwired, notify does not raise (does not block folder CRUD) but returns
   False — a wiring bug must be loud in the log, not silent.
3. A callback raising → swallowed and logged, returns False (following
   folder.py's fallback semantics: a tool-side sync failure never blocks the
   business operation).
4. renamed = unregister old then register new; if unregistering old fails, still
   attempt to register new (the tool side converges to the new state as far as
   possible).
"""

from unittest.mock import MagicMock

import pytest

from src.adapter import folder_events as fe


@pytest.fixture(autouse=True)
def _clean_hooks():
    fe.clear_folder_tool_hooks()
    yield
    fe.clear_folder_tool_hooks()


def _install():
    reg, unreg = MagicMock(return_value=True), MagicMock()
    fe.set_folder_tool_hooks(register=reg, unregister=unreg)
    return reg, unreg


def test_created_dispatches_to_register():
    reg, unreg = _install()
    assert fe.notify_folder_created("f1", "desc", "tokA") is True
    reg.assert_called_once_with("f1", "desc", "tokA")
    unreg.assert_not_called()


def test_deleted_dispatches_to_unregister():
    reg, unreg = _install()
    assert fe.notify_folder_deleted("f1", "tokA") is True
    unreg.assert_called_once_with("f1", user_token="tokA")
    reg.assert_not_called()


def test_renamed_unregisters_old_then_registers_new():
    reg, unreg = _install()
    calls = []
    unreg.side_effect = lambda *a, **k: calls.append("unreg")
    reg.side_effect = lambda *a, **k: calls.append("reg")
    assert fe.notify_folder_renamed("old", "new", "d2", "tokA") is True
    assert calls == ["unreg", "reg"], "必須先卸舊再掛新"
    unreg.assert_called_once_with("old", user_token="tokA")
    reg.assert_called_once_with("new", "d2", "tokA")


def test_renamed_still_registers_new_when_unregister_fails():
    """A failure unregistering old must not block registering new — the tool side converges to the new state as far as possible."""
    reg, unreg = _install()
    unreg.side_effect = RuntimeError("boom")
    assert fe.notify_folder_renamed("old", "new", "d", "tokA") is False  # a failure occurred → False
    reg.assert_called_once_with("new", "d", "tokA")


def test_unwired_does_not_raise_returns_false():
    """Unwired (wiring bug): does not raise, returns False — folder CRUD is not blocked."""
    assert fe.notify_folder_created("f", "d", "t") is False
    assert fe.notify_folder_deleted("f", "t") is False
    assert fe.notify_folder_renamed("a", "b", "d", "t") is False


def test_callback_exception_swallowed():
    """A callback raising → swallowed, returns False, not propagated (following the original fallback semantics)."""
    reg, _ = _install()
    reg.side_effect = RuntimeError("mcp down")
    assert fe.notify_folder_created("f", "d", "t") is False


def test_deleted_callback_exception_swallowed():
    """deleted's unregister raising: swallowed, returns False, does not block the delete flow."""
    _, unreg = _install()
    unreg.side_effect = RuntimeError("mcp down")
    assert fe.notify_folder_deleted("f", "t") is False


def test_renamed_register_failure_returns_false():
    """renamed: unregister old succeeds, register new fails → returns False (the caller must know the tool side did not converge to the new state)."""
    reg, unreg = _install()
    reg.side_effect = RuntimeError("register down")
    assert fe.notify_folder_renamed("old", "new", "d", "t") is False
    unreg.assert_called_once()
