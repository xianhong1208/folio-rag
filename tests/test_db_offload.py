
"""Regression test -- event-loop offload of blocking DB helpers.

Synchronous SQLAlchemy helpers block the event loop, so under high concurrency
API/SSE/cancel/query all stutter together. _run_db pushes such calls onto a
worker thread. Pins down that _run_db runs off the loop thread and passes
results/exceptions through unchanged.
"""

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
    assert seen["thread"] != loop_thread, "the DB helper still runs on the event-loop thread (not offloaded)"


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


# ---- adapters: sync DB helpers never run on the event-loop thread ------------------

def _recording(result):
    seen = []

    def _fn(*args, **kwargs):
        seen.append(threading.get_ident())
        return result
    return _fn, seen


async def test_run_db_alias_is_the_shared_helper():
    from src.adapter.db_offload import run_db
    from src.adapter import rag_indexing
    assert rag_indexing._run_db is run_db


async def test_folder_adapter_get_folder_offloads():
    from unittest.mock import patch
    from src.adapter.folder import FolderAdapter
    fn, seen = _recording([])
    with patch("src.adapter.folder.FolderDB.get", side_effect=fn):
        assert await FolderAdapter.get_folder(user_token="tok", id=1) == []
    assert seen and seen[0] != threading.get_ident()


async def test_file_adapter_get_file_offloads_ownership_and_query():
    from unittest.mock import MagicMock, patch
    from src.adapter.file import FileAdapter
    owner, seen_owner = _recording([MagicMock()])
    files, seen_files = _recording([])
    with patch("src.adapter.file.FolderDB.get", side_effect=owner), \
         patch("src.adapter.file.FileDB.get", side_effect=files):
        assert await FileAdapter.get_file(user_token="tok", folder_id=1) == []
    loop_thread = threading.get_ident()
    assert seen_owner[0] != loop_thread and seen_files[0] != loop_thread


async def test_maintenance_get_indexed_files_offloads():
    from unittest.mock import MagicMock, patch
    from src.adapter.rag_maintenance import RAGMaintenanceService
    fn, seen = _recording([])
    ctx = MagicMock()
    ctx.indexing_service.list_indices_for_folder.side_effect = fn
    cache = MagicMock()
    cache.get.return_value = None
    with patch("src.adapter.rag_maintenance.CacheService.get_instance", return_value=cache):
        assert await RAGMaintenanceService(ctx).get_indexed_files(7) == []
    assert seen and seen[0] != threading.get_ident()
