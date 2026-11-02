"""Index operations on the job manager: restart → Interrupted, single-file cancel,
forced retry, maintenance (Rebuild-FTS) jobs and the progress-write throttle.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.domain.exceptions import ConflictError
from src.domain.rag.index_job_manager import (
    INTERRUPTED_MESSAGE,
    JOB_KIND_REBUILD_FTS,
    IndexJobState,
    IndexingJobManager,
    JobStatus,
)


@pytest.fixture()
def manager():
    with patch("db.indexjobdb.IndexJobDB.upsert"), \
         patch("db.indexjobdb.IndexJobDB.ensure_table"), \
         patch("db.fileindexdb.FileIndexDB.ensure_content_hash_columns"):
        m = IndexingJobManager()
        m._persisted = []
        m._persist = lambda d: m._persisted.append(d)
        yield m


async def _wait_status(m, job_id, statuses, timeout=5.0):
    st = None
    for _ in range(int(timeout / 0.02)):
        st = await m.get_status(job_id)
        if st and st["status"] in statuses:
            return st
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {statuses}: {st}")


class GateAdapter:
    """index_files stub: starts each file, then waits for the test to release it."""

    def __init__(self):
        self.release = asyncio.Event()
        self.kwargs: list = []

    async def index_files(self, *, file_ids, folder_id, token, chunk_size,
                          chunk_overlap, progress_tracker, **kw):
        self.kwargs.append(kw)
        progress_tracker.on_started(len(file_ids), False)
        progress_tracker.on_file_started(1, file_ids[0], "a.pdf")
        await self.release.wait()
        self.release.clear()
        progress_tracker.mark_completed(
            {"total_files": len(file_ids), "successful": len(file_ids), "failed": 0})


# ---- restart → Interrupted -------------------------------------------------------

class TestInterruptedOnStartup:
    def _seed(self, m, **kw):
        st = IndexJobState(job_id=kw.pop("job_id", "j1"), folder_id=7,
                           status=kw.pop("status", JobStatus.RUNNING), **kw)
        m._jobs[st.job_id] = st
        return st

    def test_running_job_becomes_interrupted_not_failed(self, manager):
        self._seed(manager)
        with patch("db.indexjobdb.IndexJobDB.load_active", return_value=[]), \
             patch("db.fileindexdb.FileIndexDB.mark_failed"), \
             patch("db.fileindexdb.FileIndexDB.get_by_file", return_value=None):
            assert manager.mark_interrupted_on_startup() == 1
        st = manager._jobs["j1"]
        assert st.status == JobStatus.INTERRUPTED
        assert st.message == INTERRUPTED_MESSAGE and st.error == "server_restart"
        assert st.completed_at

    def test_in_flight_and_unstarted_files_are_marked(self, manager):
        self._seed(manager, current_file_id="f2", scope_file_ids=["f1", "f2", "f3", "f4"],
                   file_timings=[{"file_id": "f1", "status": "success"}])
        rows = {"f3": None, "f4": SimpleNamespace(status="indexed")}
        with patch("db.indexjobdb.IndexJobDB.load_active", return_value=[]), \
             patch("db.fileindexdb.FileIndexDB.mark_failed") as mf, \
             patch("db.fileindexdb.FileIndexDB.get_by_file", side_effect=rows.get):
            manager.mark_interrupted_on_startup()
        marked = {c.args[0]: c.kwargs for c in mf.call_args_list}
        # f1 finished, f4 keeps its intact index; f2 was mid-flight, f3 never started.
        assert set(marked) == {"f2", "f3"}
        assert all(k["status"] == "interrupted" and k["folder_id"] == 7 for k in marked.values())
        assert mf.call_args_list[0].args[1] == INTERRUPTED_MESSAGE

    def test_jobs_restored_from_db_are_interrupted(self, manager):
        row = {"job_id": "j9", "folder_id": 3, "status": "pending", "kind": "index",
               "scope_file_ids": ["fa"], "file_timings": []}
        with patch("db.indexjobdb.IndexJobDB.load_active", return_value=[row]), \
             patch("db.fileindexdb.FileIndexDB.mark_failed") as mf, \
             patch("db.fileindexdb.FileIndexDB.get_by_file", return_value=None):
            assert manager.mark_interrupted_on_startup() == 1
        assert manager._jobs["j9"].status == JobStatus.INTERRUPTED
        assert mf.call_args.args[0] == "fa"

    def test_maintenance_job_marks_no_files(self, manager):
        self._seed(manager, kind=JOB_KIND_REBUILD_FTS, current_file_id="f1")
        with patch("db.indexjobdb.IndexJobDB.load_active", return_value=[]), \
             patch("db.fileindexdb.FileIndexDB.mark_failed") as mf:
            manager.mark_interrupted_on_startup()
        assert manager._jobs["j1"].status == JobStatus.INTERRUPTED
        mf.assert_not_called()

    def test_old_name_still_works(self, manager):
        assert IndexingJobManager.mark_all_running_as_restart_failed \
            is IndexingJobManager.mark_interrupted_on_startup

    def test_interrupted_is_terminal(self, manager):
        self._seed(manager, status=JobStatus.INTERRUPTED)
        manager._update_job_state("j1", status=JobStatus.SUCCEEDED)
        assert manager._jobs["j1"].status == JobStatus.INTERRUPTED


# ---- single-file cancel ------------------------------------------------------------

class TestCancelFile:
    async def test_queued_file_is_removed_from_its_batch(self, manager):
        ad = GateAdapter()
        await manager.start_indexing_files(ad, file_ids=["f1"], folder_id=7, token="t",
                                           chunk_size=None, chunk_overlap=None)
        j2 = await manager.start_indexing_files(ad, file_ids=["f2", "f3"], folder_id=7, token="t",
                                                chunk_size=None, chunk_overlap=None)
        assert await manager.cancel_file("f2", 7) == "dequeued"
        assert manager._pending_file_batches[7][0].file_ids == ["f3"]
        assert (await manager.get_status(j2["job_id"]))["scope_file_ids"] == ["f3"]
        ad.release.set()
        await _wait_status(manager, j2["job_id"], {"running"})
        ad.release.set()
        await _wait_status(manager, j2["job_id"], {"succeeded"})

    async def test_last_queued_file_cancels_the_batch(self, manager):
        ad = GateAdapter()
        j1 = await manager.start_indexing_files(ad, file_ids=["f1"], folder_id=7, token="t",
                                                chunk_size=None, chunk_overlap=None)
        j2 = await manager.start_indexing_files(ad, file_ids=["f2"], folder_id=7, token="t",
                                                chunk_size=None, chunk_overlap=None)
        assert await manager.cancel_file("f2", 7) == "dequeued"
        assert (await manager.get_status(j2["job_id"]))["status"] == "cancelled"
        ad.release.set()
        await _wait_status(manager, j1["job_id"], {"succeeded"})
        assert len(ad.kwargs) == 1  # the emptied batch never ran

    async def test_sole_file_of_running_job_cancels_job(self, manager):
        ad = GateAdapter()
        j1 = await manager.start_indexing_files(ad, file_ids=["f1"], folder_id=7, token="t",
                                                chunk_size=None, chunk_overlap=None)
        await _wait_status(manager, j1["job_id"], {"running"})
        await asyncio.sleep(0.01)
        assert await manager.cancel_file("f1", 7) == "cancelled_job"
        st = await manager.get_status(j1["job_id"])
        assert st["status"] == "cancelled" and st["message"] == "Cancelled by user"

    async def test_file_in_multi_file_job_is_flagged_then_recorded_cancelled(self, manager):
        ad = GateAdapter()
        j1 = await manager.start_indexing_files(ad, file_ids=["f1", "f2"], folder_id=7, token="t",
                                                chunk_size=None, chunk_overlap=None)
        await _wait_status(manager, j1["job_id"], {"running"})
        await asyncio.sleep(0.01)
        assert await manager.cancel_file("f1", 7) == "flagged"
        assert manager.is_file_abort_requested("f1")

        written = threading.Event()
        with patch("db.fileindexdb.FileIndexDB.mark_failed",
                   side_effect=lambda *a, **k: written.set()) as mf:
            from src.domain.rag.index_job_manager import IndexProgressTracker
            IndexProgressTracker(manager, j1["job_id"]).record_file_timing(
                {"file_id": "f1", "status": "failed", "total_ms": 5})
            assert written.wait(2)
        assert mf.call_args.args[:2] == ("f1", "Cancelled by user")
        assert mf.call_args.kwargs == {"folder_id": 7, "status": "cancelled"}
        assert not manager.is_file_abort_requested("f1")
        ad.release.set()
        await _wait_status(manager, j1["job_id"], {"succeeded"})

    async def test_flagged_file_that_already_succeeded_keeps_its_index(self, manager):
        ad = GateAdapter()
        j1 = await manager.start_indexing_files(ad, file_ids=["f1", "f2"], folder_id=7, token="t",
                                                chunk_size=None, chunk_overlap=None)
        await _wait_status(manager, j1["job_id"], {"running"})
        await asyncio.sleep(0.01)
        await manager.cancel_file("f2", 7)
        with patch("db.fileindexdb.FileIndexDB.mark_failed") as mf:
            from src.domain.rag.index_job_manager import IndexProgressTracker
            IndexProgressTracker(manager, j1["job_id"]).record_file_timing(
                {"file_id": "f2", "status": "success", "total_ms": 5})
            manager._persist_executor.submit(lambda: None).result(2)
        mf.assert_not_called()
        ad.release.set()
        await _wait_status(manager, j1["job_id"], {"succeeded"})

    async def test_unknown_file_returns_none(self, manager):
        assert await manager.cancel_file("nope", 7) is None

    async def test_stale_flag_cleared_when_file_is_indexed_again(self, manager):
        manager._file_abort_flags["f1"] = True
        ad = GateAdapter()
        j = await manager.start_indexing_files(ad, file_ids=["f1"], folder_id=7, token="t",
                                               chunk_size=None, chunk_overlap=None)
        assert not manager.is_file_abort_requested("f1")
        ad.release.set()
        await _wait_status(manager, j["job_id"], {"succeeded"})

    async def test_flags_for_unreached_files_die_with_the_job(self, manager):
        ad = GateAdapter()
        j1 = await manager.start_indexing_files(ad, file_ids=["f1", "f2"], folder_id=7, token="t",
                                                chunk_size=None, chunk_overlap=None)
        await _wait_status(manager, j1["job_id"], {"running"})
        await asyncio.sleep(0.01)
        await manager.cancel_file("f2", 7)
        ad.release.set()
        await _wait_status(manager, j1["job_id"], {"succeeded"})
        await asyncio.sleep(0.02)
        assert not manager.is_file_abort_requested("f2")
        assert "f2" not in manager._cancelled_files


class TestActiveStages:
    async def test_queued_scope_files_are_reported(self, manager):
        ad = GateAdapter()
        j1 = await manager.start_indexing_files(ad, file_ids=["f1", "f2"], folder_id=7, token="t",
                                                chunk_size=None, chunk_overlap=None)
        await _wait_status(manager, j1["job_id"], {"running"})
        await asyncio.sleep(0.01)
        j2 = await manager.start_indexing_files(ad, file_ids=["f3"], folder_id=7, token="t",
                                                chunk_size=None, chunk_overlap=None)
        stages = manager.active_file_stages()
        assert stages["f2"]["stage"] == "queued" and stages["f2"]["job_id"] == j1["job_id"]
        assert stages["f3"]["stage"] == "queued" and stages["f3"]["job_id"] == j2["job_id"]
        assert stages["f1"]["job_id"] == j1["job_id"] and stages["f1"]["stage"] != "queued"
        ad.release.set()
        await _wait_status(manager, j1["job_id"], {"succeeded"})
        ad.release.set()
        await _wait_status(manager, j2["job_id"], {"succeeded"})


# ---- forced retry ------------------------------------------------------------------

class TestForce:
    async def test_force_is_forwarded_to_the_adapter(self, manager):
        ad = GateAdapter()
        j = await manager.start_indexing_files(ad, file_ids=["f1"], folder_id=7, token="t",
                                               chunk_size=None, chunk_overlap=None, force=True)
        ad.release.set()
        await _wait_status(manager, j["job_id"], {"succeeded"})
        assert ad.kwargs == [{"force": True}]

    async def test_force_survives_the_queue(self, manager):
        ad = GateAdapter()
        j1 = await manager.start_indexing_files(ad, file_ids=["f1"], folder_id=7, token="t",
                                                chunk_size=None, chunk_overlap=None)
        j2 = await manager.start_indexing_files(ad, file_ids=["f2"], folder_id=7, token="t",
                                                chunk_size=None, chunk_overlap=None, force=True)
        ad.release.set()
        await _wait_status(manager, j1["job_id"], {"succeeded"})
        await _wait_status(manager, j2["job_id"], {"running"})
        ad.release.set()
        await _wait_status(manager, j2["job_id"], {"succeeded"})
        assert ad.kwargs == [{}, {"force": True}]  # plain uploads don't send force


# ---- maintenance jobs (Rebuild FTS) ------------------------------------------------

class TestMaintenanceJob:
    async def test_runs_in_thread_with_progress_and_summary(self, manager):
        seen_threads = []

        def work(progress, should_stop):
            seen_threads.append(threading.current_thread() is threading.main_thread())
            progress(0, 2)
            progress(1, 2, "f1")
            progress(2, 2, "f2")
            return {"message": "Rebuilt 10 chunk(s)", "rows_updated": 10}

        job = await manager.start_maintenance_job(folder_id=7, kind=JOB_KIND_REBUILD_FTS, work=work)
        assert job["kind"] == JOB_KIND_REBUILD_FTS
        st = await _wait_status(manager, job["job_id"], {"succeeded"})
        assert seen_threads == [False]
        assert st["processed_files"] == 2 and st["total_files"] == 2
        assert st["message"] == "Rebuilt 10 chunk(s)" and st["result_summary"]["rows_updated"] == 10
        assert 7 not in manager._active_folders

    async def test_holds_the_folder_lock(self, manager):
        gate = threading.Event()
        job = await manager.start_maintenance_job(
            folder_id=7, kind=JOB_KIND_REBUILD_FTS, work=lambda p, s: gate.wait(5) and {})
        with pytest.raises(ConflictError):
            await manager.start_maintenance_job(folder_id=7, kind=JOB_KIND_REBUILD_FTS,
                                                work=lambda p, s: {})
        with pytest.raises(ConflictError):
            await manager.start_indexing(MagicMock(), folder_id=7, token="t", chunk_size=None,
                                         chunk_overlap=None, skip_existing=True)
        gate.set()
        await _wait_status(manager, job["job_id"], {"succeeded"})

    async def test_cancel_makes_should_stop_true(self, manager):
        started, stop_seen = threading.Event(), threading.Event()

        def work(progress, should_stop):
            started.set()
            for _ in range(500):
                if should_stop():
                    stop_seen.set()
                    return {"message": "stopped"}
                threading.Event().wait(0.01)
            return {}

        job = await manager.start_maintenance_job(folder_id=7, kind=JOB_KIND_REBUILD_FTS, work=work)
        await asyncio.get_running_loop().run_in_executor(None, started.wait, 2)
        assert await manager.cancel_job(job["job_id"])
        assert await asyncio.get_running_loop().run_in_executor(None, stop_seen.wait, 2)
        assert (await manager.get_status(job["job_id"]))["status"] == "cancelled"

    async def test_failure_marks_job_failed(self, manager):
        def boom(progress, should_stop):
            raise RuntimeError("db down")
        job = await manager.start_maintenance_job(folder_id=7, kind=JOB_KIND_REBUILD_FTS, work=boom)
        st = await _wait_status(manager, job["job_id"], {"failed"})
        assert st["error"] == "db down"
        assert 7 not in manager._active_folders

    async def test_heartbeat_while_work_runs(self, manager):
        gate = threading.Event()
        manager._MAINT_HEARTBEAT_SECONDS = 0.02
        job = await manager.start_maintenance_job(
            folder_id=7, kind=JOB_KIND_REBUILD_FTS, work=lambda p, s: gate.wait(5) and {})
        await _wait_status(manager, job["job_id"], {"running"})
        before = len(manager._persisted)
        await asyncio.sleep(0.15)
        assert len(manager._persisted) > before  # heartbeats keep the watchdog away
        gate.set()
        await _wait_status(manager, job["job_id"], {"succeeded"})


# ---- progress-write throttle (job write amplification) -----------------------------

class TestPersistThrottle:
    def _job(self, m):
        m._jobs["j1"] = IndexJobState(job_id="j1", folder_id=7, status=JobStatus.RUNNING)

    def test_progress_ticks_are_throttled(self, manager):
        self._job(manager)
        for i in range(50):
            manager._update_job_state("j1", current_file_stage="embedding",
                                      current_file_chunks_done=i, current_file_chunks_total=50)
        assert len(manager._persisted) == 1

    def test_progress_ticks_still_reach_sse(self, manager):
        self._job(manager)
        seen = []
        manager._notify_subscribers = lambda jid, d: seen.append(d["current_file_chunks_done"])
        for i in range(5):
            manager._update_job_state("j1", current_file_chunks_done=i)
        assert seen[-4:] == [1, 2, 3, 4]

    def test_non_progress_updates_always_persist(self, manager):
        self._job(manager)
        manager._update_job_state("j1", current_file_chunks_done=1)
        manager._update_job_state("j1", current_file_id="f2", current_index=2)
        manager._update_job_state("j1", current_file_chunks_done=2)
        manager._update_job_state("j1", status=JobStatus.SUCCEEDED)
        assert [d["status"] for d in manager._persisted] == ["running", "running", "succeeded"]

    def test_progress_persists_again_after_interval(self, manager):
        self._job(manager)
        manager._update_job_state("j1", current_file_chunks_done=1)
        manager._last_persist["j1"] -= manager._PERSIST_MIN_INTERVAL + 1
        manager._update_job_state("j1", current_file_chunks_done=2)
        assert len(manager._persisted) == 2
