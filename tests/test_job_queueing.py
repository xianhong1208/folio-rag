
"""Regression tests for the job identity of a queued upload.

Uploading while a folder was busy returned someone else's job_id in the
response, and after the drain the real job took yet another new id — the
frontend saw the "job record disappear / get overwritten". Deleting and
immediately re-uploading the same file reproduces it most easily (hitting the
previous job's wind-down window).

Fixed contract: a queued upload has its own PENDING job record **the moment it
enters the queue**, and its job_id stays fixed through response → queued →
RUNNING → terminal; a job cancelled while queued is skipped automatically at
dequeue.
"""

import asyncio

import pytest
from unittest.mock import patch

from src.domain.rag.index_job_manager import IndexingJobManager, JobStatus


class SlowAdapter:
    """Stub adapter with controllable completion timing."""

    def __init__(self):
        self.release = asyncio.Event()
        self.calls: list = []

    async def index_files(self, *, file_ids, folder_id, token, chunk_size,
                          chunk_overlap, progress_tracker):
        self.calls.append(list(file_ids))
        progress_tracker.on_started(len(file_ids), False)
        await self.release.wait()
        self.release.clear()
        progress_tracker.mark_completed(
            {"total_files": len(file_ids), "successful": len(file_ids), "failed": 0})


@pytest.fixture()
def manager():
    with patch("db.indexjobdb.IndexJobDB.upsert"), \
         patch("db.indexjobdb.IndexJobDB.ensure_table"), \
         patch("db.fileindexdb.FileIndexDB.ensure_content_hash_columns"), \
         patch.object(IndexingJobManager, "_persist", lambda self, d: None):
        yield IndexingJobManager()


async def _wait_status(m, job_id, statuses, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        st = await m.get_status(job_id)
        if st and st["status"] in statuses:
            return st
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} 沒到達 {statuses}: {st}")


async def test_queued_upload_gets_own_stable_job_id(manager):
    """A queued upload immediately has its own job_id, unchanged for its whole lifecycle (TC-job-queue-01)."""
    ad = SlowAdapter()
    j1 = await manager.start_indexing_files(
        ad, file_ids=["f1"], folder_id=7, token="t", chunk_size=None, chunk_overlap=None)

    # Upload again while the folder is busy → must get its "own" new job_id, not j1's
    j2 = await manager.start_indexing_files(
        ad, file_ids=["f2"], folder_id=7, token="t", chunk_size=None, chunk_overlap=None)
    assert j2["job_id"] != j1["job_id"], "排隊上傳不能回別人的 job_id(舊 bug)"
    assert j2["status"] == "pending" and j2["queued"] is True

    # The queued job appears in the list immediately (the frontend can see "queued")
    listed = {j["job_id"] for j in manager.list_jobs(folder_id=7)}
    assert j2["job_id"] in listed

    # The previous job completes → the queued job starts and completes under "the same id"
    ad.release.set()
    await _wait_status(manager, j1["job_id"], {"succeeded"})
    await _wait_status(manager, j2["job_id"], {"running"})
    assert ad.calls[-1] == ["f2"], "dequeue 啟動的就是排隊那批檔案"
    ad.release.set()
    await _wait_status(manager, j2["job_id"], {"succeeded"})


async def test_queued_job_started_at_reset_on_launch(manager):
    """A queued job's started_at is reset at actual launch, excluding the queue wait (TC-job-queue-03).

    Previously started_at was fixed at enqueue and unchanged after dequeue, so
    the frontend counted the queue wait as elapsed time. Now started_at is the
    moment execution actually begins.
    """
    ad = SlowAdapter()
    j1 = await manager.start_indexing_files(
        ad, file_ids=["x"], folder_id=9, token="t", chunk_size=None, chunk_overlap=None)
    j2 = await manager.start_indexing_files(
        ad, file_ids=["y"], folder_id=9, token="t", chunk_size=None, chunk_overlap=None)
    assert j2["queued"] is True

    # started_at while queued (= the moment it entered the queue)
    queued_started_at = (await manager.get_status(j2["job_id"]))["started_at"]

    # Give the queue wait a measurable interval (ISO8601 UTC strings compare lexicographically)
    await asyncio.sleep(0.05)

    # j1 completes → j2 is dequeued and actually launches
    ad.release.set()
    await _wait_status(manager, j1["job_id"], {"succeeded"})
    await _wait_status(manager, j2["job_id"], {"running"})

    running_started_at = (await manager.get_status(j2["job_id"]))["started_at"]
    assert running_started_at > queued_started_at, (
        "排隊 job 啟動時 started_at 應重設為啟動時刻,不該還停在入佇列時刻"
    )
    ad.release.set()
    await _wait_status(manager, j2["job_id"], {"succeeded"})


async def test_stale_cleanup_reentry_does_not_drain_queue(manager):
    """A late cleanup from a non-owner must not drain the queue / seize the folder lock (TC-job-queue-04).

    Scenario: after the watchdog forcibly reclaims job1's lock and dequeues job2,
    job1's wedged thread finally finishes and its finally block calls
    _cleanup_job_slot(job1) a second time. job1 is no longer the lock owner — if
    the drain section had no ownership guard, it would also launch queued job3
    and seize the lock, so job2 and job3 would concurrently write the same table.
    """
    ad = SlowAdapter()
    j1 = await manager.start_indexing_files(
        ad, file_ids=["a"], folder_id=11, token="t", chunk_size=None, chunk_overlap=None)
    j2 = await manager.start_indexing_files(
        ad, file_ids=["b"], folder_id=11, token="t", chunk_size=None, chunk_overlap=None)
    assert j2["queued"] is True
    # Wait until j1 actually starts (a yield point after create_task is needed before calls records ["a"])
    await _wait_status(manager, j1["job_id"], {"running"})

    # Simulate a late re-entry from an "already reclaimed job": the caller is no longer the lock owner
    manager._cleanup_job_slot("stale-ghost-job-id", 11)

    # The lock still belongs to j1, j2 stays quietly queued, and no new job is launched
    assert manager._active_folders.get(11) == j1["job_id"], "非持有者不得動 folder lock"
    assert ad.calls == [["a"]], "非持有者的 cleanup 不得啟動排隊 job"
    assert (await manager.get_status(j2["job_id"]))["status"] == "pending"

    # The guard must not break the normal chain: j1 completes → j2 is dequeued and runs under the same id as usual
    ad.release.set()
    await _wait_status(manager, j1["job_id"], {"succeeded"})
    await _wait_status(manager, j2["job_id"], {"running"})
    ad.release.set()
    await _wait_status(manager, j2["job_id"], {"succeeded"})


async def test_cancelled_queued_job_is_skipped_at_dequeue(manager):
    """A job cancelled while queued is skipped automatically at dequeue, moving to the next (TC-job-queue-02)."""
    ad = SlowAdapter()
    j1 = await manager.start_indexing_files(
        ad, file_ids=["a"], folder_id=8, token="t", chunk_size=None, chunk_overlap=None)
    j2 = await manager.start_indexing_files(
        ad, file_ids=["b"], folder_id=8, token="t", chunk_size=None, chunk_overlap=None)
    j3 = await manager.start_indexing_files(
        ad, file_ids=["c"], folder_id=8, token="t", chunk_size=None, chunk_overlap=None)

    # Cancel j2 while queued (corresponds to "delete the file right after upload")
    assert await manager.cancel_job(j2["job_id"]) is True
    st2 = await manager.get_status(j2["job_id"])
    assert st2["status"] == "cancelled"

    # j1 completes → skip j2 and launch j3 directly (same id)
    ad.release.set()
    await _wait_status(manager, j1["job_id"], {"succeeded"})
    await _wait_status(manager, j3["job_id"], {"running"})
    assert ad.calls[-1] == ["c"]
    ad.release.set()
    await _wait_status(manager, j3["job_id"], {"succeeded"})
    # j2 stays cancelled, not revived by dequeue
    assert (await manager.get_status(j2["job_id"]))["status"] == "cancelled"
