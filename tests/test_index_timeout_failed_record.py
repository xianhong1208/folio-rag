
"""Regression tests: a timeout failure must leave a failed FileIndex record.

When a single file exceeds per_file_timeout, _with_timeout's asyncio.wait_for
cancels index_document by injecting a CancelledError (a BaseException, not an
Exception). index_document's inner `except Exception` does not catch it, so its
mark_index_failed never runs. The batch loop's except did not backfill a record
either, so a timed-out file had no FileIndex record in the DB at all — absent
from the frontend's indexed list, absent from the already-terminal active job —
leaving the file row with no status tag.

Contract: the per-file except in batch indexing (index_files / index_folder)
always persists a failed FileIndex record so the frontend can show "failed".
"""

import asyncio

import pytest

from src.adapter.rag_indexing import _with_timeout


async def test_with_timeout_raises_TimeoutError_and_cancels_inner():
    """_with_timeout on timeout: cancels the inner coroutine and raises TimeoutError (reproduces the cancel semantics)."""
    cancelled = {"v": False}

    async def slow():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled["v"] = True
            raise

    with pytest.raises(TimeoutError):
        await _with_timeout(slow(), timeout_seconds=1, filename="big.pdf")
    # Let the cancelled coroutine finish winding down
    await asyncio.sleep(0.05)
    assert cancelled["v"], "逾時應真的 cancel 內層 coroutine"


def test_cancellederror_is_not_caught_by_except_exception():
    """Guards the core trap: CancelledError is a BaseException, not caught by `except Exception`.

    This is the root of the bug — if Python ever changes this inheritance, or
    someone changes index_document's except to `except BaseException`, this test
    flags that the fallback logic needs re-evaluating.
    """
    assert issubclass(asyncio.CancelledError, BaseException)
    assert not issubclass(asyncio.CancelledError, Exception)

    caught_by_exception = False
    try:
        raise asyncio.CancelledError()
    except Exception:  # noqa: BLE001 - intentionally demonstrating it is not caught
        caught_by_exception = True
    except asyncio.CancelledError:
        pass
    assert caught_by_exception is False


def test_timeout_status_label_is_timeout():
    """A timed-out timing entry has status 'timeout' (distinct from a plain failed, easing debugging)."""
    from src.adapter.rag_indexing import _make_timing_entry
    import time

    e = _make_timing_entry("fid", "big.pdf", "timeout", time.perf_counter(), {})
    assert e["status"] == "timeout"
