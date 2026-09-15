
"""Background indexing job manager

Provides utilities to run folder indexing in background tasks and track progress.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Set

from cachetools import TTLCache

from src.config.model import IndexingConfig  # single source for the timeout default
from src.domain.exceptions import ConflictError
from src.log import get_api_logger

logger = get_api_logger()


class JobStatus(str, Enum):
    """Index job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL_SUCCESS = "partial_success"   # some files indexed, some failed
    FAILED = "failed"
    CANCELLED = "cancelled"


# Terminal statuses that release the per-folder lock and allow cleanup
_TERMINAL_STATUSES = {
    JobStatus.SUCCEEDED,
    JobStatus.PARTIAL_SUCCESS,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}


@dataclass
class _PendingFileBatch:
    """A file-indexing request that arrived while the folder was busy.

    Each batch owns its own job record (PENDING status, stable job_id) as soon
    as it is queued — visible in the upload response and the /index/jobs list
    from the first moment. When the previous job finishes, _cleanup_job_slot
    launches the next still-PENDING job in the queue (cancelled ones are
    skipped automatically); the job_id never changes.
    """

    job_id: str
    adapter: Any
    file_ids: List[str]
    folder_id: int
    token: str
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]


@dataclass
class IndexJobState:
    """Runtime state for a background indexing job."""

    job_id: str
    folder_id: int
    status: JobStatus = JobStatus.PENDING
    total_files: int = 0
    processed_files: int = 0
    current_index: int = 0
    current_file_id: Optional[str] = None
    current_file_name: Optional[str] = None
    last_file_status: Optional[str] = None
    last_message: Optional[str] = None
    message: str = "Waiting to start"
    error: Optional[str] = None
    skip_existing: bool = True
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    # file_ids handled by this job; populated only for file-scoped jobs, empty for folder-wide
    scope_file_ids: List[str] = field(default_factory=list)
    # Heartbeat watchdog: refreshed on every _update_job_state call.
    last_updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # per-file timing: {file_id, file_name, status, total_ms, load_ms, index_ms, chunks}
    file_timings: List[Dict[str, Any]] = field(default_factory=list)
    # Chunk-level progress (current file only). In-memory + SSE live view, not persisted —
    # reset when the file completes; on restart the job is marked failed, so stale values are meaningless.
    # stage: loading (docling parse) / contextualizing (LLM prefix) / embedding / writing (pgvector insert)
    current_file_stage: Optional[str] = None
    current_file_chunks_done: Optional[int] = None
    current_file_chunks_total: Optional[int] = None
    # ETA (seconds) — same as chunk progress: in-memory + SSE live view, not persisted.
    # current_file_eta_seconds: remaining seconds for the current stage, extrapolated from the
    #   measured rate (None right after a stage switch or before any rate sample; UI shows "estimating").
    # eta_seconds: coarse estimate for the whole job = avg time per completed file × remaining files
    #   − elapsed on the current file (None before the first file completes).
    current_file_eta_seconds: Optional[int] = None
    eta_seconds: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable copy of the job state."""
        data = asdict(self)
        data["status"] = self.status.value
        return data


class IndexProgressTracker:
    """Callback helper used by RAGAdapter while indexing."""

    def __init__(self, manager: "IndexingJobManager", job_id: str):
        self._manager = manager
        self._job_id = job_id

    def on_started(self, total_files: int, skip_existing: bool):
        self._manager._update_job_state(
            self._job_id,
            total_files=total_files,
            skip_existing=skip_existing,
            message="Indexing started" if total_files else "Preparing indexing",
            status=JobStatus.RUNNING,
            processed_files=0,
            current_index=0,
            current_file_id=None,
            current_file_name=None,
        )

    def on_file_started(
        self,
        current_index: int,
        file_id: str,
        file_name: str,
    ):
        self._manager._note_file_started(self._job_id)
        self._manager._update_job_state(
            self._job_id,
            current_index=current_index,
            current_file_id=file_id,
            current_file_name=file_name,
            message=f"Indexing file {current_index}: {file_name}",
            current_file_stage=None,
            current_file_chunks_done=None,
            current_file_chunks_total=None,
            current_file_eta_seconds=None,
        )

    def on_chunk_progress(
        self,
        stage: str,
        done: Optional[int],
        total: Optional[int],
    ):
        """Chunk-level progress (called after indexer throttling: context-gen every 10 chunks, embedding per batch).

        Args:
            stage: loading / contextualizing / embedding / writing.
            done: chunks completed in this stage.
            total: total chunks for this file.
        """
        # Compute ETA alongside the progress update (stage-rate extrapolation + coarse job estimate)
        self._manager._update_chunk_progress(self._job_id, stage, done, total)

    def on_file_completed(
        self,
        processed_files: int,
        status: str,
        message: str,
    ):
        self._manager._update_job_state(
            self._job_id,
            processed_files=processed_files,
            last_file_status=status,
            last_message=message,
            current_file_stage=None,
            current_file_chunks_done=None,
            current_file_chunks_total=None,
            current_file_eta_seconds=None,
        )

    def record_file_timing(self, entry: Dict[str, Any]) -> None:
        """Append one per-file timing entry to the job state (called by the indexing service after each file).

        The list is part of result_summary so consumers can rank the slowest files without parsing logs.

        Args:
            entry: ``{file_id, file_name, status, total_ms, load_ms, index_ms}``.
        """
        with self._manager._lock:
            state = self._manager._jobs.get(self._job_id)
            if not state:
                return
            # Same terminal gate as _update_job_state: drop timings that arrive after the job is terminal
            if state.status in _TERMINAL_STATUSES:
                return
            state.file_timings.append(entry)
            # A file just completed = best point to refresh the job ETA (between files, current elapsed=None)
            state.eta_seconds = self._manager._compute_job_eta_locked(
                state, current_file_elapsed=None
            )
            state.last_updated_at = datetime.now(timezone.utc).isoformat()
            snapshot = state.to_dict()
        # file_timings is a write-heavy field; persist outside the lock
        self._manager._persist(snapshot)

    def mark_completed(self, summary: Dict[str, Any]):
        # Terminal status: all succeeded → SUCCEEDED; some failed → PARTIAL_SUCCESS; 0 succeeded with ≥1 failed → FAILED
        successful = summary.get("successful", 0) or 0
        failed = summary.get("failed", 0) or 0
        if failed > 0 and successful == 0:
            status = JobStatus.FAILED
        elif failed > 0:
            status = JobStatus.PARTIAL_SUCCESS
        else:
            status = JobStatus.SUCCEEDED

        self._manager._update_job_state(
            self._job_id,
            status=status,
            message=summary.get("message", "Indexing completed"),
            processed_files=summary.get("total_files", 0),
            current_index=summary.get("total_files", 0),
            current_file_id=None,
            current_file_name=None,
            result_summary=summary,
            completed_at=datetime.now(timezone.utc).isoformat(),
            eta_seconds=0,
            current_file_eta_seconds=None,
        )

    def mark_failed(self, error: str):
        self._manager._update_job_state(
            self._job_id,
            status=JobStatus.FAILED,
            error=error,
            message="Indexing failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )


class IndexingJobManager:
    """Singleton manager coordinating background indexing jobs."""

    _instance: Optional["IndexingJobManager"] = None

    # Job state retention — completed jobs stay queryable for 24h then evict.
    # Caps total entries at 10k to bound memory if many indexing jobs fire.
    _JOB_TTL_SECONDS = 86400
    _JOB_MAX_ENTRIES = 10000

    # Watchdog tunables — scan every 60s, stale after 10min without heartbeat.
    # This threshold detects a genuinely dead job, not a per-file duration cap (that is the per-file
    # wait_for). A slow file in a heartbeat blind spot would be wrongly cancelled here, so
    # rag_indexing._ProgressKeepalive emits a heartbeat every 60s to cover all blind spots; confirm
    # that coverage before lowering this or adding a blocking path that bypasses progress_cb.
    _WATCHDOG_INTERVAL_SECONDS = 60
    _HEARTBEAT_STALE_SECONDS = 600
    # Last-resort folder-lock reclaim. The lock's only normal exit is _cleanup_job_slot (task
    # finally); an uninterruptible blocking call (docling/GPU hang) can skip it and lock the folder
    # forever. If a terminal job's task has not exited past this grace period, force-reclaim (accepting
    # the residual-write risk). 1h is far longer than any legitimate blocking call, so it only catches
    # a genuinely wedged task.
    _LOCK_RECLAIM_GRACE_SECONDS = 3600

    # Fallback job-level cap — only used if config can't be read at all. References the IndexingConfig
    # field default (10 days) instead of duplicating the 864000 literal.
    # Resolution order: env RAG_JOB_TIMEOUT_SECONDS > config.rag.indexing.job_timeout_seconds > this.
    _JOB_TIMEOUT_FALLBACK = IndexingConfig.model_fields["job_timeout_seconds"].default

    # Upper bound cancel_job waits for a task to actually exit. Blocking calls inside to_thread
    # (embedding batch / bulk insert) can't be interrupted — we can only wait for the step to finish;
    # 30s covers the worst case for a single batch.
    _CANCEL_WAIT_SECONDS = 30

    def __init__(self):
        # TTLCache avoids the unbounded-dict memory leak from earlier impl;
        # GET also returns None for expired keys, matching the old dict.get() contract.
        self._jobs: "TTLCache[str, IndexJobState]" = TTLCache(
            maxsize=self._JOB_MAX_ENTRIES,
            ttl=self._JOB_TTL_SECONDS,
        )
        self._lock = Lock()
        # Folder-level lock — at most one running job per folder
        self._active_folders: Dict[int, str] = {}
        # Per-folder queue of file-indexing batches that arrived during a
        # running job. Drained by _cleanup_job_slot when the running job
        # terminates, spawning a single merged follow-up job. Lets users
        # batch-upload many files without hitting "Folder already has a
        # running indexing job" and silently losing 9/10 files.
        self._pending_file_batches: Dict[int, List[_PendingFileBatch]] = {}
        # Cancellation — track the asyncio.Task for each running job
        self._task_handles: Dict[str, asyncio.Task] = {}
        # Internal tracking for ETA computation (kept out of state to avoid API noise):
        # {job_id: {"file_start": ts, "stage": str, "stage_start": ts, "baseline": int}}
        self._progress_track: Dict[str, Dict[str, Any]] = {}
        # per-file abort flags — set by FileAdapter.delete_file when a file is deleted, consumed
        # (and cleared) by the adapter's mid-stage probe. TTL is a safety net: pending files are
        # blocked by gate 1's DB probe and never consume their flag, so expiry reclaims them.
        self._file_abort_flags: "TTLCache[str, bool]" = TTLCache(
            maxsize=4096, ttl=self._JOB_TTL_SECONDS,
        )
        # Watchdog handle (idempotent start)
        self._watchdog_task: Optional[asyncio.Task] = None
        # Reference to the main event loop so worker-thread progress reports can wake SSE
        # thread-safely. Captured in start_watchdog (which always runs on the loop thread).
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Strong references to fire-and-forget background tasks (drain, etc.) — asyncio keeps only
        # a weak ref, so without this set a queued job could be garbage-collected before it runs.
        self._bg_tasks: Set[asyncio.Task] = set()
        # Single-worker executor dedicated to DB persistence: upsert is sync SQLAlchemy (SELECT+commit),
        # and running it directly on the event loop would stall the whole server (API/SSE/cancel) on
        # every progress tick. A single worker guarantees write ordering (out-of-order writes would let
        # an old snapshot overwrite a newer one).
        self._persist_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="idxjob-persist",
        )
        # Per-job SSE event subscribers. Each value is the list of live
        # asyncio.Queue subscribers waiting for state updates of that job.
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        # ensure the IndexJobs table exists; persistence is best-effort
        # (a DB failure here does not break the in-memory manager).
        try:
            from db.indexjobdb import IndexJobDB
            IndexJobDB.ensure_table()
        except Exception as e:
            logger.warning(f"IndexJobs table init skipped (will run in-memory only): {e}")
        # Piggyback on manager bootstrap to add the content_hash columns
        try:
            from db.fileindexdb import FileIndexDB
            FileIndexDB.ensure_content_hash_columns()
        except Exception as e:
            logger.warning(f"content_hash columns migration skipped: {e}")

    @classmethod
    def get_instance(cls) -> "IndexingJobManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def _resolve_job_timeout(cls) -> int:
        """Resolve the job-level timeout in seconds.

        Priority: env RAG_JOB_TIMEOUT_SECONDS > config.rag.indexing.job_timeout_seconds > fallback.

        Returns:
            Cap in seconds; <=0 means no limit.
        """
        import os
        env_val = os.getenv("RAG_JOB_TIMEOUT_SECONDS")
        if env_val:
            try:
                return int(env_val)
            except ValueError:
                logger.warning(
                    f"Invalid RAG_JOB_TIMEOUT_SECONDS={env_val!r}, using config/default"
                )
        try:
            from src.config.config_manager import Config
            cfg = Config.get_config_model()
            if cfg and cfg.rag and cfg.rag.indexing:
                return cfg.rag.indexing.job_timeout_seconds
        except Exception as e:
            logger.warning(f"Failed to read job_timeout_seconds from config: {e}")
        return cls._JOB_TIMEOUT_FALLBACK

    async def start_indexing(
        self,
        adapter,
        *,
        folder_id: int,
        token: str,
        chunk_size: Optional[int],
        chunk_overlap: Optional[int],
        skip_existing: bool,
    ) -> Dict[str, Any]:
        """Schedule folder indexing in the background."""
        job_id = str(uuid.uuid4())
        state = IndexJobState(job_id=job_id, folder_id=folder_id, skip_existing=skip_existing)

        with self._lock:
            if folder_id in self._active_folders:
                existing = self._active_folders[folder_id]
                raise ConflictError(
                    message=f"Folder {folder_id} already has a running indexing job: {existing}",
                    resource=f"indexing_job:{folder_id}",
                )
            self._jobs[job_id] = state
            self._active_folders[folder_id] = job_id

        # initial persist so the job is queryable from DB immediately
        self._persist(state.to_dict())

        tracker = IndexProgressTracker(self, job_id)

        job_timeout = self._resolve_job_timeout()

        async def _run_folder():
            try:
                core = adapter.index_folder(
                    folder_id=folder_id,
                    token=token,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    skip_existing=skip_existing,
                    progress_tracker=tracker,
                )
                # job-level hard timeout. <=0 disables the cap.
                if job_timeout and job_timeout > 0:
                    await asyncio.wait_for(core, timeout=job_timeout)
                else:
                    await core
            except asyncio.CancelledError:
                logger.info(f"Indexing job {job_id} cancelled")
                raise
            except asyncio.TimeoutError:
                logger.error(
                    f"Indexing job {job_id} exceeded job_timeout={job_timeout}s — marking failed"
                )
                tracker.mark_failed(f"Job exceeded {job_timeout}s job-level timeout")
            except Exception as exc:
                logger.error(
                    f"Indexing job {job_id} for folder {folder_id} failed: {exc}"
                )
                tracker.mark_failed(str(exc))
            finally:
                self._cleanup_job_slot(job_id, folder_id)

        # asyncio task, no thread — index_folder is already async, and running it in a thread pool
        # via asyncio.run would tie up a worker thread for the entire indexing run; 32 concurrent
        # jobs would deadlock the pool (blocking even STT/OCR's run_in_threadpool).
        task = asyncio.create_task(_run_folder())
        with self._lock:
            self._task_handles[job_id] = task

        logger.info(
            f"Scheduled indexing job {job_id} for folder {folder_id} (skip_existing={skip_existing})"
        )

        return state.to_dict()

    async def start_indexing_files(
        self,
        adapter,
        *,
        file_ids: list,
        folder_id: int,
        token: str,
        chunk_size: Optional[int],
        chunk_overlap: Optional[int],
    ) -> Dict[str, Any]:
        """Schedule indexing of specific files in the background.

        Used for auto-indexing on file upload. Only indexes the specified files,
        not the entire folder.

        Args:
            adapter: RAGAdapter instance
            file_ids: List of file IDs (UUID strings) to index
            folder_id: Target folder ID
            token: User token
            chunk_size: Chunk size (None = use default)
            chunk_overlap: Chunk overlap (None = use default)

        Returns:
            Job state dictionary
        """
        job_id = str(uuid.uuid4())
        state = IndexJobState(job_id=job_id, folder_id=folder_id, skip_existing=False)
        # Record planned scope so the per-file status endpoint can detect "queued"
        state.scope_file_ids = [str(fid) for fid in file_ids]

        # folder lock applies to file-scoped jobs too — but unlike start_indexing
        # (which rejects), we queue the batch and let it drain after the current
        # job finishes. This is the upload-N-files-in-a-row case: file 1 starts
        # a job, file 2…10 used to fail with AUTO_INDEX_FAIL. Now they queue.
        #
        # A queued upload also owns its own job record immediately (PENDING, stable job_id),
        # so the id the caller receives stays valid all the way through RUNNING to terminal.
        with self._lock:
            if folder_id in self._active_folders:
                existing = self._active_folders[folder_id]
                state.message = f"Queued behind job {existing}"
                self._jobs[job_id] = state
                self._pending_file_batches.setdefault(folder_id, []).append(
                    _PendingFileBatch(
                        job_id=job_id,
                        adapter=adapter,
                        file_ids=[str(fid) for fid in file_ids],
                        folder_id=folder_id,
                        token=token,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
                pending_total = sum(
                    len(b.file_ids) for b in self._pending_file_batches[folder_id]
                )
                queued = True
            else:
                self._jobs[job_id] = state
                self._active_folders[folder_id] = job_id
                queued = False

        if queued:
            self._persist(state.to_dict())
            logger.info(
                f"Folder {folder_id} busy with job {existing}; queued job {job_id} "
                f"({len(file_ids)} file(s), total pending: {pending_total})"
            )
            # Keep the legacy "queued": True for the caller to display; job_id is this job's own
            # and stays the same all the way through RUNNING → terminal.
            return {**state.to_dict(), "queued": True}

        self._launch_file_job(
            state, adapter,
            file_ids=[str(fid) for fid in file_ids],
            folder_id=folder_id, token=token,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        return state.to_dict()

    def _launch_file_job(
        self,
        state: IndexJobState,
        adapter,
        *,
        file_ids: List[str],
        folder_id: int,
        token: str,
        chunk_size: Optional[int],
        chunk_overlap: Optional[int],
    ) -> None:
        """Launch a file-scoped job (the caller must have already assigned the folder lock to it).

        Shared by the immediate path in start_indexing_files and the dequeue path in
        _cleanup_job_slot, so jobs started via either path behave identically
        (timeout / cancel / cleanup / drain). Must be called on the event loop (create_task).
        """
        job_id = state.job_id
        # started_at = the moment execution actually begins, not the enqueue moment. A queued job
        # builds its state at enqueue time (started_at=then) but may not start until it is dequeued
        # minutes later — resetting here keeps queue wait time out of the displayed elapsed time.
        # On the immediate path the reset is effectively a no-op (state was just created). The
        # watchdog uses last_updated_at for liveness and ETA uses per-file averages; neither looks
        # at started_at, so the reset has no side effects.
        state.started_at = datetime.now(timezone.utc).isoformat()
        self._persist(state.to_dict())
        tracker = IndexProgressTracker(self, job_id)
        job_timeout = self._resolve_job_timeout()

        async def _run_files():
            try:
                core = adapter.index_files(
                    file_ids=file_ids,
                    folder_id=folder_id,
                    token=token,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    progress_tracker=tracker,
                )
                # job-level hard timeout. <=0 disables the cap.
                if job_timeout and job_timeout > 0:
                    await asyncio.wait_for(core, timeout=job_timeout)
                else:
                    await core
            except asyncio.CancelledError:
                logger.info(f"File indexing job {job_id} cancelled")
                raise
            except asyncio.TimeoutError:
                logger.error(
                    f"File indexing job {job_id} exceeded job_timeout={job_timeout}s — marking failed"
                )
                tracker.mark_failed(f"Job exceeded {job_timeout}s job-level timeout")
            except Exception as exc:
                logger.error(
                    f"File indexing job {job_id} for folder {folder_id} failed: {exc}"
                )
                tracker.mark_failed(str(exc))
            finally:
                self._cleanup_job_slot(job_id, folder_id)

        task = asyncio.create_task(_run_files())
        with self._lock:
            self._task_handles[job_id] = task

        logger.info(
            f"Scheduled file indexing job {job_id} for folder {folder_id} with {len(file_ids)} files"
        )

    def _cleanup_job_slot(self, job_id: str, folder_id: int):
        """Release the task_handle + folder lock slot after a job ends (idempotent); drain the pending queue.

        This is the ONLY release point for the folder lock (reached only when the task truly exits).
        _update_job_state does not release the lock when flipping to a terminal status — the task may
        still be stuck in a blocking to_thread call, and releasing early would let a new job write to
        the same table as a residual write.

        After releasing the lock, dequeue the next still-PENDING batch for this folder and launch its
        pre-created job (same job_id) — queued jobs cancelled in the meantime are skipped. Only one is
        dequeued at a time (the folder lock serialises naturally); when that job ends its own cleanup
        dequeues the next, draining the whole queue in a chain.

        Args:
            job_id: the job to clean up.
            folder_id: its folder.
        """
        next_batch: Optional[_PendingFileBatch] = None
        with self._lock:
            self._task_handles.pop(job_id, None)
            self._progress_track.pop(job_id, None)
            # Ownership guard: only the job that actually holds the folder lock may release it,
            # drain the queue, and hand the lock to the next job. After the watchdog force-reclaims,
            # the original wedged task may wake up later and re-enter here via its finally — it is no
            # longer the holder, and without this guard it would launch the next queued job and seize
            # the lock, running two jobs writing the same table concurrently (breaking the "at most one
            # running job per folder" invariant). A non-owner only clears its own task_handle /
            # progress_track (idempotent, harmless).
            owned = self._active_folders.get(folder_id) == job_id
            if owned:
                self._active_folders.pop(folder_id, None)

                queue = self._pending_file_batches.get(folder_id)
                while queue:
                    cand = queue.pop(0)
                    cand_state = self._jobs.get(cand.job_id)
                    if cand_state is not None and cand_state.status == JobStatus.PENDING:
                        next_batch = cand
                        break
                    # Cancelled (CANCELLED) or evaporated by TTL while queued → skip and keep looking
                    logger.info(
                        f"Folder {folder_id}: skipping dequeued job {cand.job_id} "
                        f"(status={getattr(cand_state, 'status', 'evicted')})"
                    )
                if not queue:
                    self._pending_file_batches.pop(folder_id, None)
                if next_batch is not None:
                    # Hand the folder lock to the next job while still holding the lock — the gap
                    # before create_task must not let an incoming start_indexing steal the slot.
                    self._active_folders[folder_id] = next_batch.job_id
            else:
                logger.warning(
                    f"Folder {folder_id}: cleanup from non-owner job {job_id} "
                    f"(current owner: {self._active_folders.get(folder_id)}) — "
                    f"skipping lock release / queue drain (late reentry after reclaim?)"
                )

        if next_batch is not None:
            state = self._jobs.get(next_batch.job_id)
            logger.info(
                f"Folder {folder_id}: dequeuing job {next_batch.job_id} "
                f"({len(next_batch.file_ids)} file(s))"
            )
            try:
                self._update_job_state(next_batch.job_id, message="Dequeued — starting")
                self._launch_file_job(
                    state, next_batch.adapter,
                    file_ids=next_batch.file_ids,
                    folder_id=folder_id, token=next_batch.token,
                    chunk_size=next_batch.chunk_size,
                    chunk_overlap=next_batch.chunk_overlap,
                )
            except RuntimeError:
                # No running loop (e.g. cleanup ran from a non-async context).
                # Release the slot just taken and mark the job FAILED; don't leave a permanent PENDING ghost.
                with self._lock:
                    if self._active_folders.get(folder_id) == next_batch.job_id:
                        self._active_folders.pop(folder_id, None)
                self._update_job_state(
                    next_batch.job_id,
                    status=JobStatus.FAILED,
                    error="could not schedule dequeued job (no asyncio loop)",
                )
                logger.error(
                    f"Folder {folder_id}: could not schedule dequeued job "
                    f"{next_batch.job_id} (no asyncio loop)."
                )

    async def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return current status for a job."""
        with self._lock:
            state = self._jobs.get(job_id)
            return state.to_dict() if state else None

    def list_jobs(
        self,
        status_filter: Optional[JobStatus] = None,
        folder_id: Optional[int] = None,
    ) -> list[Dict[str, Any]]:
        """List jobs in the in-memory cache.

        Args:
            status_filter: return only jobs with this status (RUNNING / PENDING / SUCCEEDED / FAILED / CANCELLED)
            folder_id: return only jobs for this folder
        """
        with self._lock:
            jobs = []
            for state in self._jobs.values():
                if status_filter is not None and state.status != status_filter:
                    continue
                if folder_id is not None and state.folder_id != folder_id:
                    continue
                jobs.append(state.to_dict())
            return jobs

    def active_file_stages(self) -> Dict[str, Dict[str, Any]]:
        """Return {file_id: {stage, done, total, eta}} — which stage each currently-indexing file is in.

        Used by the admin file list to show "parsing / contextualizing / embedding / writing" live.
        Only includes the current file of RUNNING jobs; in-memory, not persisted.
        """
        out: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for state in self._jobs.values():
                if state.status != JobStatus.RUNNING or not state.current_file_id:
                    continue
                out[str(state.current_file_id)] = {
                    "stage": state.current_file_stage,
                    "done": state.current_file_chunks_done,
                    "total": state.current_file_chunks_total,
                    "eta": state.current_file_eta_seconds,
                }
        return out

    def count_by_status(self) -> Dict[str, int]:
        """Return {pending: N, running: N, succeeded: N, failed: N, cancelled: N, total: N}.

        TTLCache evicts expired entries automatically, so the succeeded/failed counts are for the
        recent TTL window, not all-time.
        """
        counts = {s.value: 0 for s in JobStatus}
        with self._lock:
            for state in self._jobs.values():
                counts[state.status.value] += 1
            counts["total"] = sum(counts.values())
        return counts

    async def cancel_jobs_for_folder(self, folder_id: int) -> list[str]:
        """Cancel all PENDING/RUNNING jobs for the folder.

        Call before destructive operations (delete folder / delete index) so no in-flight job writes
        to a state that is being torn down.

        Args:
            folder_id: the folder to clear.

        Returns:
            List of cancelled job_ids; an empty list means nothing was running.
        """
        with self._lock:
            # Drop queued batches first — otherwise a cancelled job's _cleanup_job_slot would drain
            # them into a new job that collides with the destructive operation (DROP TABLE).
            dropped = self._pending_file_batches.pop(folder_id, None)
            targets = [
                state.job_id
                for state in self._jobs.values()
                if state.folder_id == folder_id
                and state.status in (JobStatus.PENDING, JobStatus.RUNNING)
            ]
        if dropped:
            n_files = sum(len(b.file_ids) for b in dropped)
            logger.info(
                f"Dropped {n_files} queued file(s) for folder {folder_id} "
                f"(destructive operation pending)"
            )
        cancelled = []
        for job_id in targets:
            try:
                if await self.cancel_job(job_id):
                    cancelled.append(job_id)
            except Exception as e:
                logger.warning(f"cancel_jobs_for_folder: cancel of {job_id} failed: {e}")
        if cancelled:
            logger.info(
                f"Cancelled {len(cancelled)} running job(s) for folder {folder_id}: {cancelled}"
            )
        return cancelled

    def purge_jobs_for_folder(self, folder_id: int) -> int:
        """Completely remove all job records for the folder from the in-memory cache + DB.

        Called on folder deletion (cancel_jobs_for_folder first, then purge). Without this, orphan
        jobs linger in the /index/jobs list and the frontend keeps polling their folder_id for a
        folder that no longer exists → endless FOLDER_NOT_FOUND log spam.

        Args:
            folder_id: the deleted folder id.

        Returns:
            Number of jobs removed from the in-memory cache.
        """
        with self._lock:
            doomed = [
                job_id for job_id, state in self._jobs.items()
                if state.folder_id == folder_id
            ]
            for job_id in doomed:
                self._jobs.pop(job_id, None)
                self._task_handles.pop(job_id, None)
                self._progress_track.pop(job_id, None)
            self._pending_file_batches.pop(folder_id, None)
            if self._active_folders.get(folder_id) in doomed:
                self._active_folders.pop(folder_id, None)
        # Clear the DB side too (best-effort; failures only log, they don't block folder deletion)
        try:
            from db.indexjobdb import IndexJobDB
            IndexJobDB.delete_for_folder(folder_id)
        except Exception as e:
            logger.warning(f"IndexJob DB purge for folder {folder_id} failed: {e}")
        if doomed:
            logger.info(f"Purged {len(doomed)} job record(s) for deleted folder {folder_id}")
        return len(doomed)

    async def cancel_job(self, job_id: str, reason: str = "Cancelled by user") -> bool:
        """Cancel a running or queued job.

        Args:
            job_id: identifier of the job to cancel.
            reason: cancellation reason written into the job state (shown in the frontend job status).

        Returns:
            True on successful cancel; False if not found or already terminal.

        Note: this method waits until the task actually finishes (up to _CANCEL_WAIT_SECONDS) before
        returning. Blocking calls inside to_thread are not interrupted by cancel(); without the wait,
        a caller (delete folder → DROP TABLE) would race the residual insert, and PGVectorStore.add
        would even recreate the dropped table, leaving an orphaned table behind.
        """
        with self._lock:
            state = self._jobs.get(job_id)
            if not state or state.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                return False
            task = self._task_handles.get(job_id)
        if task and not task.done():
            task.cancel()
        self._update_job_state(
            job_id,
            status=JobStatus.CANCELLED,
            message=reason,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        if task and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self._CANCEL_WAIT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Job {job_id} task still running {self._CANCEL_WAIT_SECONDS}s after "
                    f"cancel (blocking call in flight); caller proceeds at its own risk"
                )
            except asyncio.CancelledError:
                pass  # task ended with CancelledError = normal cancellation
            except Exception:
                pass  # the task's own exceptions are handled inside _run_*; here we only care whether it stopped
        return True

    async def abort_indexing_for_file(
        self, file_id: str, folder_id: Optional[int] = None,
    ) -> Optional[str]:
        """Called when a file is deleted: abort that file's in-flight indexing as fast as possible.

        Two-tier strategy:
        - Single-file job (scope is exactly this file) → cancel the whole job (preemptive;
          task.cancel() takes effect at the next await point).
        - Multi-file / folder-wide job → only set the per-file abort flag, which the adapter's
          mid-stage probe (cooperative) uses to abort just that file at its next checkpoint; other
          files are unaffected.
        Pending files not yet reached need no action — the adapter's gate 1 blocks them via a DB probe.

        Args:
            file_id: the deleted file's UUID (str).
            folder_id: pass when known to narrow the scan.

        Returns:
            ``"cancelled_job:<job_id>"`` / ``"flagged"`` / None (no relevant job).
        """
        fid = str(file_id)
        with self._lock:
            target: Optional[IndexJobState] = None
            for state in self._jobs.values():
                if state.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                    continue
                if folder_id is not None and state.folder_id != folder_id:
                    continue
                if state.current_file_id == fid or fid in state.scope_file_ids:
                    target = state
                    break
        if target is None:
            return None

        if target.scope_file_ids == [fid]:
            if await self.cancel_job(
                target.job_id, reason="Cancelled: file deleted during indexing",
            ):
                logger.info(
                    f"Cancelled single-file indexing job {target.job_id} — "
                    f"file {fid} was deleted"
                )
                return f"cancelled_job:{target.job_id}"

        with self._lock:
            self._file_abort_flags[fid] = True
        logger.info(
            f"Flagged file {fid} for indexing abort (job {target.job_id}); "
            f"mid-stage probe will stop it at the next checkpoint"
        )
        return "flagged"

    def is_file_abort_requested(self, file_id: str) -> bool:
        """Used by the adapter's mid-stage probe: whether this file has been requested to abort."""
        with self._lock:
            return str(file_id) in self._file_abort_flags

    def clear_file_abort(self, file_id: str) -> None:
        """Clear the flag once it has been consumed (the file's indexing has been aborted)."""
        with self._lock:
            self._file_abort_flags.pop(str(file_id), None)

    async def _watchdog_loop(self):
        """Background poll: scan RUNNING jobs every 60s and mark any with no heartbeat for >10min as FAILED."""
        while True:
            await asyncio.sleep(self._WATCHDOG_INTERVAL_SECONDS)
            try:
                now = datetime.now(timezone.utc)
                with self._lock:
                    stale = []
                    for state in list(self._jobs.values()):
                        if state.status != JobStatus.RUNNING:
                            continue
                        try:
                            last = datetime.fromisoformat(state.last_updated_at)
                        except ValueError:
                            continue
                        if (now - last).total_seconds() > self._HEARTBEAT_STALE_SECONDS:
                            stale.append(state.job_id)
                for job_id in stale:
                    logger.warning(
                        f"Heartbeat lost for job {job_id} — marking failed"
                    )
                    self._update_job_state(
                        job_id,
                        status=JobStatus.FAILED,
                        error="Heartbeat lost (>10min without update)",
                        completed_at=datetime.now(timezone.utc).isoformat(),
                    )
                    # Flipping to FAILED is only bookkeeping; the task must actually be cancelled
                    # (the terminal gate blocks write-back). The folder lock is not released here —
                    # _cleanup_job_slot releases it when the task truly exits, so even if the task is
                    # stuck in an uninterruptible blocking call, no new job can enter and two jobs
                    # never write the same table.
                    with self._lock:
                        task = self._task_handles.get(job_id)
                    if task and not task.done():
                        task.cancel()
                        logger.warning(f"Cancelled stale task for job {job_id}")

                # Last-resort reclaim of a folder lock wedged because the job is terminal but the
                # lock-holding task has not exited. Normally the lock is released by _cleanup_job_slot
                # (task finally); when the task is stuck in an uninterruptible blocking call the finally
                # never runs and the folder would be locked forever.
                reclaim: List[tuple] = []
                with self._lock:
                    for folder_id, holder in list(self._active_folders.items()):
                        state = self._jobs.get(holder)
                        task = self._task_handles.get(holder)
                        if state is None:
                            # holder was evicted by TTL but still holds the lock → overdue for reclaim
                            reclaim.append((folder_id, holder, "holder evicted from cache"))
                            continue
                        if state.status not in _TERMINAL_STATUSES:
                            continue  # running / queued, leave alone
                        if task is None or task.done():
                            continue  # task already exited; _cleanup will/did release the lock, don't force
                        try:
                            done_at = (
                                datetime.fromisoformat(state.completed_at)
                                if state.completed_at else None
                            )
                        except ValueError:
                            done_at = None
                        if done_at and (now - done_at).total_seconds() > self._LOCK_RECLAIM_GRACE_SECONDS:
                            reclaim.append((folder_id, holder, "terminal task wedged past grace"))
                for folder_id, holder, why in reclaim:
                    logger.error(
                        f"Force-reclaiming folder lock {folder_id} from job {holder} "
                        f"({why}); wedged task may still be alive — residual-write risk"
                    )
                    self._cleanup_job_slot(holder, folder_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Watchdog tick failed (will retry next interval)")

    def start_watchdog(self):
        """Start the watchdog background task (idempotent; repeated calls start it only once)."""
        if self._watchdog_task is not None and not self._watchdog_task.done():
            return
        self._loop = asyncio.get_running_loop()  # capture the main loop for cross-thread wakeups
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        logger.info(
            f"Job watchdog started (interval={self._WATCHDOG_INTERVAL_SECONDS}s, "
            f"stale_after={self._HEARTBEAT_STALE_SECONDS}s)"
        )

    def mark_all_running_as_restart_failed(self) -> int:
        """Flip all PENDING/RUNNING jobs left in the DB to FAILED (reason=server_restart).

        Called on startup: orphan jobs left by a dead previous process are no longer running, so close
        them out. Idempotent; safe to call multiple times.

        Returns:
            Number of jobs flipped to failed.
        """
        flipped = 0
        # 1. Load DB jobs that were left mid-flight by a previous process
        try:
            from db.indexjobdb import IndexJobDB
            for row in IndexJobDB.load_active():
                job_id = str(row.get("job_id"))
                with self._lock:
                    if job_id in self._jobs:
                        # already in memory — _update_job_state below will handle it
                        continue
                    state = IndexJobState(
                        job_id=job_id,
                        folder_id=row.get("folder_id"),
                        status=JobStatus(row.get("status", "running")),
                        total_files=row.get("total_files") or 0,
                        processed_files=row.get("processed_files") or 0,
                        current_index=row.get("current_index") or 0,
                        current_file_id=row.get("current_file_id"),
                        current_file_name=row.get("current_file_name"),
                        last_file_status=row.get("last_file_status"),
                        last_message=row.get("last_message"),
                        message=row.get("message") or "",
                        error=row.get("error"),
                        skip_existing=row.get("skip_existing", True),
                        started_at=row.get("started_at") or datetime.now(timezone.utc).isoformat(),
                        completed_at=row.get("completed_at"),
                        last_updated_at=row.get("last_updated_at") or datetime.now(timezone.utc).isoformat(),
                        result_summary=row.get("result_summary"),
                        scope_file_ids=row.get("scope_file_ids") or [],
                        file_timings=row.get("file_timings") or [],
                    )
                    self._jobs[job_id] = state
        except Exception as e:
            logger.warning(f"A4 DB scan skipped (will only process in-memory jobs): {e}")

        # 2. Flip everything still PENDING/RUNNING in the cache
        with self._lock:
            job_ids = [
                state.job_id
                for state in self._jobs.values()
                if state.status in (JobStatus.PENDING, JobStatus.RUNNING)
            ]
        for job_id in job_ids:
            self._update_job_state(
                job_id,
                status=JobStatus.FAILED,
                error="server_restart",
                message="Job marked failed due to server restart",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            flipped += 1
        if flipped:
            logger.info(f"Marked {flipped} stale running jobs as failed (server_restart)")
        return flipped

    # ETA — the "time remaining" shown next to chunk progress

    def _note_file_started(self, job_id: str) -> None:
        """Record the current file's start time; reset stage tracking (per-file)."""
        with self._lock:
            self._progress_track[job_id] = {"file_start": time.time()}

    @staticmethod
    def _compute_job_eta_locked(state: IndexJobState, current_file_elapsed: Optional[float]) -> Optional[int]:
        """Coarse estimate of remaining seconds for the whole job. Caller must hold self._lock.

        Method: avg time per completed file (excluding skipped) × remaining files, minus the time
        already spent on the current file (if any). Deliberately coarse but honest: returns None
        before the first file completes rather than inventing a number.

        Args:
            state: job state (reads file_timings / total_files / processed_files).
            current_file_elapsed: seconds already spent on the current file; pass None between files.

        Returns:
            Estimated remaining seconds (≥0) or None (insufficient samples).
        """
        durations_ms = [
            t.get("total_ms") for t in state.file_timings
            if t.get("status") != "skipped" and t.get("total_ms")
        ]
        if not durations_ms:
            return None
        avg_s = (sum(durations_ms) / len(durations_ms)) / 1000.0
        remaining = max(state.total_files - state.processed_files, 0)
        if remaining <= 0:
            return 0
        eta = remaining * avg_s
        if current_file_elapsed is not None:
            # The current file is part of remaining; subtract the time already spent (never let ETA go negative)
            eta -= min(current_file_elapsed, avg_s)
        return int(max(eta, 0))

    def _update_chunk_progress(
        self,
        job_id: str,
        stage: str,
        done: Optional[int],
        total: Optional[int],
    ) -> None:
        """Update chunk progress + both ETA levels together (implementation of tracker.on_chunk_progress).

        Stage ETA: within one stage, estimate the rate from (done − baseline)/elapsed and extrapolate;
        None right after a stage switch or when elapsed < 1s (too few samples, the estimate jumps around).
        """
        now = time.time()
        stage_eta: Optional[int] = None
        job_eta: Optional[int] = None
        with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return
            track = self._progress_track.setdefault(job_id, {"file_start": now})
            if track.get("stage") != stage:
                # Stage switch: reset the rate baseline (stages have completely different rates, don't mix them)
                track["stage"] = stage
                track["stage_start"] = now
                track["baseline"] = done or 0
            elif done is not None and total:
                elapsed = now - track["stage_start"]
                progressed = done - track.get("baseline", 0)
                if elapsed >= 1.0 and progressed > 0:
                    rate = progressed / elapsed
                    stage_eta = int(max(total - done, 0) / rate)
            job_eta = self._compute_job_eta_locked(
                state, current_file_elapsed=now - track.get("file_start", now)
            )
        self._update_job_state(
            job_id,
            current_file_stage=stage,
            current_file_chunks_done=done,
            current_file_chunks_total=total,
            current_file_eta_seconds=stage_eta,
            eta_seconds=job_eta,
        )

    def _update_job_state(self, job_id: str, **updates):
        with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                logger.warning(f"Attempted to update unknown job_id={job_id}")
                return

            # One-way terminal gate: once a job is SUCCEEDED/FAILED/CANCELLED/PARTIAL, any late update
            # is rejected. Without it, after the watchdog flips a job to FAILED the original task could
            # finish and flip it back to SUCCEEDED; after cancel_job sets CANCELLED, a dying task's
            # mark_failed would overwrite the cancellation reason.
            if state.status in _TERMINAL_STATUSES:
                logger.debug(
                    f"Ignoring update to terminal job {job_id} "
                    f"(status={state.status.value}, attempted={updates.get('status')})"
                )
                return

            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
            # every state change is a heartbeat
            state.last_updated_at = datetime.now(timezone.utc).isoformat()
            # Rewriting renews the TTL. TTLCache's ttl counts from the write (GET does not renew it),
            # while job_timeout can reach 10 days >> _JOB_TTL_SECONDS (24h). Without a rewrite on every
            # heartbeat, a long RUNNING job would be evicted at 24h and later updates would become
            # "unknown job_id". Terminal jobs early-exit via the gate above and are reclaimed normally
            # after 24h. Rewriting the same key doesn't grow the cache, so nothing else is evicted.
            self._jobs[job_id] = state
            if updates.get("status") in _TERMINAL_STATUSES and state.completed_at is None:
                state.completed_at = datetime.now(timezone.utc).isoformat()

            # The folder lock is not released here — its only exit is _cleanup_job_slot (when the task
            # truly exits). A terminal status does not mean the task stopped: it may still be in an
            # uninterruptible blocking to_thread call, and releasing early would let a new job's write
            # race the residual write.

            # write-through persist (snapshot inside the lock to avoid races)
            snapshot = state.to_dict()
        self._persist(snapshot)

    def _persist(self, state_dict: Dict[str, Any]) -> None:
        """best-effort write-through to IndexJobs table. Never raises.

        upsert is sync SQLAlchemy, dispatched to the single-worker executor — progress ticks are
        frequent (context-gen every 10 chunks, embedding per batch) and running them directly on the
        event loop would stall API/SSE/cancel. The manager's memory is the source of truth and the DB
        is a passive projection, so persisting tens of ms late is fine; a single worker guarantees
        upsert ordering.
        """
        try:
            self._persist_executor.submit(self._persist_sync, state_dict)
        except Exception as e:
            logger.warning(f"IndexJob persist scheduling failed (continuing in-memory): {e}")
        # SSE consumes the in-memory snapshot (canonical) directly, without waiting for the DB write
        self._notify_subscribers(state_dict.get("job_id"), state_dict)

    @staticmethod
    def _persist_sync(state_dict: Dict[str, Any]) -> None:
        """Executor worker: the actual DB upsert (IndexJobDB swallows exceptions and only logs)."""
        try:
            from db.indexjobdb import IndexJobDB
            IndexJobDB.upsert(state_dict)
        except Exception as e:
            logger.warning(f"IndexJob persist failed (continuing in-memory): {e}")

    # SSE subscriber fan-out

    def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe to a job's state changes (for the SSE endpoint).

        Args:
            job_id: the job to subscribe to.

        Returns:
            A bounded asyncio.Queue; the full state dict is put on it on every state change.
            The caller must call unsubscribe() when done, or the manager will leak the reference.
        """
        # Bounded so a slow consumer cannot OOM the server. 100 updates is
        # plenty for any sane job (we emit ~3-5 events per file processed).
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        """Remove a subscriber queue. Idempotent.

        Args:
            job_id: the corresponding job.
            q: the same queue previously returned by subscribe().
        """
        with self._lock:
            subs = self._subscribers.get(job_id)
            if not subs:
                return
            try:
                subs.remove(q)
            except ValueError:
                pass
            if not subs:
                self._subscribers.pop(job_id, None)

    def _notify_subscribers(self, job_id: Optional[str], state_dict: Dict[str, Any]) -> None:
        """Push the latest state to all subscribers (best-effort; a slow consumer is dropped without error).

        Args:
            job_id: the job to notify; None → return immediately.
            state_dict: the full state dict (same as get_status returns).
        """
        if not job_id:
            return
        with self._lock:
            subs = list(self._subscribers.get(job_id, ()))
            loop = self._loop
        if not subs:
            return

        def _deliver():
            for q in subs:
                try:
                    q.put_nowait(state_dict)
                except asyncio.QueueFull:
                    pass
                except Exception as e:
                    logger.warning(f"SSE notify dropped for job {job_id}: {e}")

        # asyncio.Queue is not thread-safe; put_nowait's internal consumer wakeup uses call_soon
        # (not threadsafe). This method can be called from the docling worker thread (page-level
        # progress reports via to_thread), and putting directly could lose an SSE wakeup. If not on
        # the loop thread, marshal the delivery back onto the loop.
        try:
            asyncio.get_running_loop()
            on_loop = True
        except RuntimeError:
            on_loop = False
        if on_loop:
            _deliver()
        elif loop is not None:
            loop.call_soon_threadsafe(_deliver)
        else:
            _deliver()  # extreme fallback (loop not captured); no worse than the original behaviour
