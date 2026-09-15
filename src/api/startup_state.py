
"""Startup-time capability load state.

app.py loads routers / MCP tools / modules under a broad try/except; failures are only logged and
the server still starts -> the endpoint silently 404s while /health reports healthy. This registers
load failures so /health/detailed reflects which capabilities are down, making capability loss visible.

Module-level singleton (written once at startup, read-only thereafter), with zero external dependencies.
"""

from typing import Dict

_LOAD_FAILURES: Dict[str, str] = {}


def record_load_failure(name: str, error) -> None:
    """Register a load failure. error may be an exception object (its type name is stored, never the full message) or a string."""
    _LOAD_FAILURES[name] = (
        type(error).__name__ if isinstance(error, BaseException) else str(error)
    )


def get_load_failures() -> Dict[str, str]:
    """Return a copy of the load failures (name -> reason). An empty dict means everything loaded successfully."""
    return dict(_LOAD_FAILURES)


def clear_load_failures() -> None:
    """Clear all entries (for tests; not called during normal operation)."""
    _LOAD_FAILURES.clear()
