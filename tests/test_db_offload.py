
"""Regression test -- event-loop offload of blocking DB helpers.

Synchronous SQLAlchemy helpers block the event loop, so under high concurrency
API/SSE/cancel/query all stutter together. _run_db pushes such calls onto a
worker thread. Pins down that _run_db runs off the loop thread and passes
results/exceptions through unchanged.
"""

import asyncio
import threading

import pytest

from src.adapter.rag_indexing import _run_db


async def test_run_db_executes_off_loop_thread():
    """_run_db must run on a different thread -- this is the only proof of "not blocking the event loop"."""
    loop_thread = threading.get_ident()
    seen = {}

    def _blocking():
        seen["thread"] = threading.get_ident()
        return "ok"

    result = await _run_db(_blocking)
    assert result == "ok"
    assert seen["thread"] != loop_thread, "DB helper 仍在 event loop 執行緒上跑(沒 offload)"


async def test_run_db_passes_args_and_returns():
    """Positional + keyword arguments are passed through unchanged, and the return value comes back unchanged."""
    def _add(a, b, *, c=0):
        return a + b + c

    assert await _run_db(_add, 2, 3, c=5) == 10


async def test_run_db_propagates_exception():
    """An exception raised by the helper must propagate up unchanged (so the caller can catch it and use the existing error handling)."""
    def _boom():
        raise ValueError("db exploded")

    with pytest.raises(ValueError, match="db exploded"):
        await _run_db(_boom)
