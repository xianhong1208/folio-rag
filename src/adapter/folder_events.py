
"""Folder event hooks — dependency inversion for MCP tool registration.

Each folder has a dedicated MCP query tool, so folder create/rename/delete must
register, swap, or unregister it. Calling the fastmcp layer directly from the
adapter created an import cycle, so app.py injects the register/unregister
callables via set_folder_tool_hooks() at startup and folder.py only calls the
notify_* functions here.

A tool-surface sync failure never blocks folder CRUD: a raising callback is
logged and returns False; missing wiring is logged at ERROR but does not raise.
"""

from __future__ import annotations

from typing import Callable, Optional

from src.log import get_adapter_logger

logger = get_adapter_logger()

# Injected by app.py at startup; None means not wired (a startup bug, logged at
# ERROR when a notify_* is called).
_register_tool: Optional[Callable] = None    # (folder_name, description, user_token) -> bool
_unregister_tool: Optional[Callable] = None  # (folder_name, *, user_token) -> None


def set_folder_tool_hooks(*, register: Callable, unregister: Callable) -> None:
    """Inject the MCP layer's register/unregister callables at app startup (must run before any folder CRUD)."""
    global _register_tool, _unregister_tool
    _register_tool = register
    _unregister_tool = unregister
    logger.info("[INIT] Folder tool hooks wired (register/unregister)")


def clear_folder_tool_hooks() -> None:
    """Clear the hooks (for test isolation; not called during normal runtime)."""
    global _register_tool, _unregister_tool
    _register_tool = None
    _unregister_tool = None


def _unwired(action: str) -> bool:
    # A startup wiring bug (set_folder_tool_hooks was never called): log loudly
    # for ops, but do not raise or block folder CRUD.
    logger.error(
        f"folder_events.{action}: tool hooks NOT wired — the MCP tool surface will not sync. "
        f"app.py should call set_folder_tool_hooks() at startup"
    )
    return False


def notify_folder_created(folder_name: str, description: Optional[str], user_token: str) -> bool:
    """Register a folder's dedicated MCP tool after creation. Does not raise on failure; returns False."""
    if _register_tool is None:
        return _unwired("created")
    try:
        _register_tool(folder_name, description, user_token)
        return True
    except Exception as e:
        logger.warning(f"Failed to register MCP tool for folder {folder_name}: {e}")
        return False


def notify_folder_deleted(folder_name: str, user_token: str) -> bool:
    """Unregister a folder's MCP tool before deletion. Does not raise on failure; returns False."""
    if _unregister_tool is None:
        return _unwired("deleted")
    try:
        _unregister_tool(folder_name, user_token=user_token)
        return True
    except Exception as e:
        logger.warning(f"Failed to unregister MCP tool for folder {folder_name}: {e}")
        return False


def notify_folder_renamed(
    old_name: str, new_name: str, description: Optional[str], user_token: str
) -> bool:
    """Refresh a folder's MCP tool after a rename or description change.

    Unregisters the old name then registers the new; a failed unregister still
    attempts the register so the tool surface converges. Returns False on failure.
    """
    if _register_tool is None or _unregister_tool is None:
        return _unwired("renamed")
    ok = True
    try:
        _unregister_tool(old_name, user_token=user_token)
    except Exception as e:
        logger.warning(f"Failed to unregister old MCP tool '{old_name}': {e}")
        ok = False
    try:
        _register_tool(new_name, description, user_token)
    except Exception as e:
        logger.warning(f"Failed to register MCP tool for folder {new_name}: {e}")
        ok = False
    return ok
