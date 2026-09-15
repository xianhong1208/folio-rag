"""RAG indexing service: single-file, folder, file-id subset, and auto-index.

Shares the indexer / indexing_service / vector_store_manager provided by RAGContext.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.api.router.response import (
    FileIndexResult,
    FileRequest,
    IndexDocumentResponse,
    IndexFolderResponse,
)
from src.config.model import IndexingConfig  # single source for the timeout default
from src.domain.exceptions import (
    DomainException,
    FileIndexingError,
    RAGFileNotFoundError,
)
from src.domain.rag.hierarchical_indexer import IndexingAbortedError
from src.log import get_adapter_logger, log_err, log_op, log_warn

if TYPE_CHECKING:
    from typing import Callable

    from src.adapter.rag_context import RAGContext
    from src.domain.rag.index_job_manager import IndexProgressTracker

logger = get_adapter_logger()

# Throttle interval (seconds) for the DB existence check in the mid-stage abort probe.
# The manager's in-memory abort flag is checked at every checkpoint; this DB probe is
# only a backstop for deletions the manager cannot notify about (e.g. a direct DB delete).
_ABORT_PROBE_INTERVAL_SECONDS = 10.0

# Keepalive tick interval (seconds). Must stay far below the watchdog's stale threshold
# (index_job_manager._HEARTBEAT_STALE_SECONDS, 600s): the watchdog fails and cancels a
# job after 600s without a heartbeat, but the per-file budget is up to 10 days, so a slow
# file in a heartbeat blind spot (semaphore wait, docling/OCR, context-gen throttle,
# embedding batch, pgvector write) would be killed prematurely. Injected heartbeats keep
# "slow" from being mistaken for "dead"; a genuinely hung file is still caught by
# _with_timeout's wait_for(per_file_timeout).
_KEEPALIVE_INTERVAL_SECONDS = 60.0


# Max concurrent indexing; env RAG_INDEXING_CONCURRENCY > config.rag.indexing.max_concurrent_jobs > 4
_INDEXING_SEMAPHORE: Optional[asyncio.Semaphore] = None

# Per-file timeout default (env > config > this). Reference the IndexingConfig field
# default directly instead of repeating the literal 864000.
_PER_FILE_TIMEOUT_DEFAULT = IndexingConfig.model_fields["per_file_timeout_seconds"].default


async def _run_db(fn, *args, **kwargs):
    """Run a blocking synchronous DB helper on a worker thread so it doesn't hold the event loop.

    baseDB is session-per-call (a fresh session created and closed each time), so
    running it on a thread is thread-safe.

    Warning: do not use inside an `except asyncio.CancelledError:` block — awaiting
    during cancellation re-raises CancelledError. Those paths (mark_index_failed on
    cancel) stay synchronous.
    """
    return await asyncio.to_thread(fn, *args, **kwargs)


def _resolve_indexing_concurrency() -> int:
    """Resolve max concurrent indexing jobs: env var > config > default 4."""
    env_val = os.getenv("RAG_INDEXING_CONCURRENCY")
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            logger.warning(f"Invalid RAG_INDEXING_CONCURRENCY={env_val!r}, using config/default")

    try:
        from src.config.config_manager import Config
        cfg = Config.get_config_model()
        if cfg and cfg.rag and cfg.rag.indexing:
            return cfg.rag.indexing.max_concurrent_jobs
    except Exception as e:
        logger.warning(f"Failed to read indexing config: {e}")

    return 4


def _resolve_per_file_timeout() -> int:
    """Resolve the per-file indexing timeout in seconds.

    Priority: env RAG_PER_FILE_TIMEOUT_SECONDS > config.rag.indexing.per_file_timeout_seconds
    > _PER_FILE_TIMEOUT_DEFAULT (864000s / 10 days). 0 or negative means no limit.
    """
    env_val = os.getenv("RAG_PER_FILE_TIMEOUT_SECONDS")
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            logger.warning(f"Invalid RAG_PER_FILE_TIMEOUT_SECONDS={env_val!r}, using config/default")
    try:
        from src.config.config_manager import Config
        cfg = Config.get_config_model()
        if cfg and cfg.rag and cfg.rag.indexing:
            return cfg.rag.indexing.per_file_timeout_seconds
    except Exception as e:
        logger.warning(f"Failed to read per_file_timeout from config: {e}")
    return _PER_FILE_TIMEOUT_DEFAULT


def _get_indexing_semaphore() -> asyncio.Semaphore:
    """Lazy-init because a Semaphore must be created after the event loop is running."""
    global _INDEXING_SEMAPHORE
    if _INDEXING_SEMAPHORE is None:
        limit = _resolve_indexing_concurrency()
        _INDEXING_SEMAPHORE = asyncio.Semaphore(limit)
        logger.info(f"[INIT] Indexing semaphore initialized (max_concurrent={limit})")
    return _INDEXING_SEMAPHORE


def _make_timing_entry(
    file_id, file_name: str, status: str, t0: float, stages: Dict[str, float],
    chunks: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble one IndexJobState.file_timings entry.

    status is one of success / failed / timeout / skipped; t0 is the perf_counter()
    taken when the per-file try block was entered. Stages that did not run are absent
    (None). chunks is the leaf-chunk count, present only on success.
    """
    return {
        "file_id": str(file_id),
        "file_name": file_name,
        "status": status,
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "load_ms": stages.get("load_ms"),
        "index_ms": stages.get("index_ms"),
        # Per-stage wall-clock from StageTimer: {loading, contextualizing, embedding, writing}
        "stage_ms": stages.get("stage_ms"),
        "chunks": chunks,
    }


async def _with_timeout(coro, timeout_seconds: int, filename: str):
    """Run a per-file indexing coroutine under a hard time budget.

    timeout_seconds <= 0 means no limit. filename is used only in the error message.
    Raises TimeoutError on timeout, carrying the filename and second count.
    """
    if timeout_seconds and timeout_seconds > 0:
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"indexing exceeded {timeout_seconds}s for file '{filename}'"
            )
    return await coro


def _keepalive_stage_label(stage: str, elapsed_seconds: float) -> str:
    """Stage label reported to the job state / SSE when the keepalive injects a heartbeat.

    Returns the stage unchanged. Do not append elapsed: the frontend badge looks the
    stage up in a localized label map keyed by the exact stage string, so any suffix
    misses the key and degrades the badge to the raw English stage. Surface elapsed via
    the free-text message / last_message fields instead. elapsed_seconds is unused for now.
    """
    return stage


class _ProgressKeepalive:
    """Wrap ``_progress_cb`` so it emits a heartbeat at least every interval seconds.

    Has the same signature as ``_progress_cb(stage, done, total)`` and can be passed to
    the indexer in its place. It remembers the last progress and, from a background task,
    re-emits it after being idle longer than the interval, refreshing last_updated_at.
    Real progress resets the idle timer, so a normally-progressing file gets no extra ticks.
    """

    def __init__(self, progress_cb, interval: float = _KEEPALIVE_INTERVAL_SECONDS):
        self._cb = progress_cb
        self._interval = interval
        # Seed a value before any stage, for heartbeats injected while waiting on the semaphore
        self._last_args = ("queued", None, None)
        self._last_emit = time.monotonic()
        self._task: Optional[asyncio.Task] = None

    def __call__(self, stage: str, done: Optional[int], total: Optional[int]) -> None:
        """Record the last progress, then forward it."""
        self._last_args = (stage, done, total)
        self._emit(stage, done, total)

    def _emit(self, stage: str, done: Optional[int], total: Optional[int]) -> None:
        self._last_emit = time.monotonic()
        if self._cb is None:
            return
        try:
            self._cb(stage, done, total)
        except Exception:
            pass  # a progress-report failure must not affect indexing

    async def _loop(self) -> None:
        while True:
            idle = self._interval - (time.monotonic() - self._last_emit)
            if idle > 0:
                await asyncio.sleep(idle)
                continue
            stage, done, total = self._last_args
            elapsed = time.monotonic() - self._last_emit
            self._emit(_keepalive_stage_label(stage, elapsed), done, total)

    async def __aenter__(self) -> "_ProgressKeepalive":
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *_exc) -> bool:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        return False


class RAGIndexingService:
    """Service handling single-file / folder / subset indexing and the auto-index trigger."""

    def __init__(self, ctx: "RAGContext"):
        self._ctx = ctx

    def _file_still_exists(self, file_id) -> bool:
        """Check whether file_id is still in the Files table (mid-job existence gate).

        Lets a job abort early when the user deletes a file mid-indexing, rather than
        burning Docling/embed and then hitting an FK violation. Fail-open: returns True
        (still present) on probe error, so a probe failure never aborts a valid job.
        """
        try:
            records = self._ctx.indexing_service.get_files_by_ids([str(file_id)])
        except Exception as e:
            logger.warning(
                f"[FILE_EXISTS_PROBE] fid={file_id} | probe failed, "
                f"assuming file still exists: {e}"
            )
            return True
        return str(file_id) in records

    def _make_should_abort(self, file_id) -> "Callable[[], bool]":
        """Build a per-file mid-stage abort probe for in-stage checkpoints beyond the gates.

        Two signal layers (True means abort): (1) IndexingJobManager's in-memory per-file
        abort flag, checked at every checkpoint and cleared on consume; (2) the
        _file_still_exists DB probe, throttled to once per _ABORT_PROBE_INTERVAL_SECONDS,
        as a backstop for deletions the manager cannot notify about. Never raises, and once
        an abort is decided it sticks (later calls return True without hitting the DB).
        """
        fid = str(file_id)
        state = {"aborted": False, "last_probe": time.monotonic()}

        def _should_abort() -> bool:
            if state["aborted"]:
                return True
            try:
                from src.domain.rag.index_job_manager import IndexingJobManager
                manager = IndexingJobManager.get_instance()
                if manager.is_file_abort_requested(fid):
                    manager.clear_file_abort(fid)
                    state["aborted"] = True
                    return True
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"[ABORT_PROBE] fid={fid} | manager flag check failed: {e}")
            now = time.monotonic()
            if now - state["last_probe"] >= _ABORT_PROBE_INTERVAL_SECONDS:
                state["last_probe"] = now
                if not self._file_still_exists(fid):  # the probe itself is fail-open
                    state["aborted"] = True
            return state["aborted"]

        return _should_abort

    async def index_document(
        self,
        file_record: FileRequest,
        token: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        folder_id_override: Optional[int] = None,
        force: bool = False,
        _timings_out: Optional[Dict[str, float]] = None,
        _progress_cb=None,
    ) -> IndexDocumentResponse:
        """Index a document into the vector store.

        chunk_size / chunk_overlap default to the config values when None. force=True
        skips the content_hash idempotency short-circuit and re-runs the whole pipeline
        (used to rebuild a single file after a config change); the stale-chunk purge
        before writing guarantees no old vectors remain. _timings_out and _progress_cb
        are internal hooks for per-file audit and background-job progress reporting.

        Raises FileIndexingError on failure.
        """
        file_id = file_record.id
        ctx = self._ctx

        # _progress_cb is None for the synchronous single-file API: no background job to
        # keep alive, so skip the extra keepalive task.
        if _progress_cb is None:
            async with _get_indexing_semaphore():
                return await self._index_document_locked(
                    file_record=file_record,
                    token=token,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    folder_id_override=folder_id_override,
                    _timings_out=_timings_out,
                    _progress_cb=None,
                )

        # The keepalive wraps outside the semaphore because waiting on the shared
        # semaphore is itself a heartbeat blind spot; wrapping inside would let the
        # watchdog misjudge that wait.
        async with _ProgressKeepalive(_progress_cb) as keepalive:
            # StageTimer sits outermost to record each stage's wall-clock time; same-stage
            # keepalive heartbeats don't trigger a switch. Result goes to file_timings.stage_ms.
            from src.domain.rag.stage_timer import StageTimer
            timer = StageTimer(inner=keepalive)
            try:
                async with _get_indexing_semaphore():
                    return await self._index_document_locked(
                        file_record=file_record,
                        token=token,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        folder_id_override=folder_id_override,
                        force=force,
                        _timings_out=_timings_out,
                        _progress_cb=timer,
                    )
            finally:
                # Finalize timing on success, failure, and timeout alike.
                if _timings_out is not None:
                    stage_ms = timer.finish()
                    if stage_ms:
                        _timings_out["stage_ms"] = stage_ms

    async def _index_document_locked(
        self,
        file_record: FileRequest,
        token: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        folder_id_override: Optional[int] = None,
        force: bool = False,
        _timings_out: Optional[Dict[str, float]] = None,
        _progress_cb=None,
    ) -> IndexDocumentResponse:
        """Index core; the caller (index_document) must hold the semaphore to enter."""
        file_id = file_record.id
        ctx = self._ctx
        # Pre-bind folder_id so the except handler can pass it to mark_index_failed even
        # if a validation error fires before it is parsed.
        folder_id: Optional[int] = None

        try:
            if not file_record:
                raise ValueError("file_record cannot be None")
            if not file_record.file_path:
                raise ValueError("file_path cannot be empty")
            if not file_record.file_name:
                raise ValueError("file_name cannot be empty")

            try:
                raw_folder_id = (
                    folder_id_override
                    if folder_id_override is not None
                    else file_record.folder_id
                )
                if raw_folder_id in (None, "None", ""):
                    raise ValueError(
                        f"folder_id is missing for file: {file_record.file_name}"
                    )
                folder_id = int(raw_folder_id)
            except (ValueError, TypeError) as ve:
                raise ValueError(
                    f"Invalid folder_id: '{raw_folder_id}' "
                    f"(type: {type(raw_folder_id).__name__}). Error: {ve}"
                )

            # Gate 1: file deleted mid-way -> abort early, saving Docling/embed/ctx-gen.
            if not await _run_db(self._file_still_exists, file_id):
                log_warn(
                    logger, "INDEX_ABORT",
                    folder_id=folder_id, file_id=file_id, token=token,
                    msg=f"file {file_id} was deleted during indexing — aborting (gate 1)",
                )
                raise RAGFileNotFoundError(file_id=str(file_id))

            # Idempotency: unchanged content_hash reuses the existing index without
            # re-embedding. force=True bypasses this (the only single-file rebuild path).
            current_hash = getattr(file_record, "content_hash", None)
            if current_hash and not force:
                existing_idx = await _run_db(ctx.indexing_service.get_index_for_file, file_id)
                if (
                    existing_idx
                    and getattr(existing_idx, "status", None) == "indexed"
                    and getattr(existing_idx, "content_hash", None) == current_hash
                ):
                    log_op(
                        logger, "INDEX_SKIP_IDEMPOTENT",
                        folder_id=existing_idx.folder_id, file_id=file_id, token=token,
                        msg=(
                            f"content_hash unchanged — reusing existing FileIndex "
                            f"({existing_idx.num_chunks} chunks)"
                        ),
                    )
                    return IndexDocumentResponse(
                        file_id=str(file_id),
                        index_id=existing_idx.index_id,
                        num_chunks=existing_idx.num_chunks or 0,
                        status="already_indexed_same_content",
                        indexed_at=(
                            existing_idx.indexed_at.isoformat()
                            if existing_idx.indexed_at else ""
                        ),
                        message="Skipped: content_hash unchanged from prior index",
                    )

            # Chunk sizes are already fixed from hierarchy_sizes in ctx; the parameters are
            # kept only for backward compatibility (IndexJobManager still passes them) and
            # recorded for audit, but ctx.leaf_chunk_size / ctx.chunk_overlap actually apply.
            try:
                if chunk_size is None or chunk_size == "":
                    chunk_size = ctx.leaf_chunk_size
                else:
                    chunk_size = int(chunk_size)
                if chunk_overlap is None or chunk_overlap == "":
                    chunk_overlap = ctx.chunk_overlap
                else:
                    chunk_overlap = int(chunk_overlap)
            except (ValueError, TypeError) as ve:
                raise ValueError(
                    f"Invalid chunk parameters. chunk_size={chunk_size}, "
                    f"chunk_overlap={chunk_overlap}. Error: {ve}"
                )

            folder = await _run_db(ctx.indexing_service.get_folder, folder_id, token)
            vector_store = ctx.get_vector_store(folder_id, token, folder=folder)
            vector_store_table = f"data_{folder_id}_{folder.vector_table_uuid}"

            # Model-switch guard: refuse to mix embedding spaces in one folder. When
            # dimensions match (e.g. bge-m3 and e5 are both 1024) the insert won't error,
            # but the spaces are incompatible and retrieval silently degrades. Fix is a reindex.
            from db.fileindexdb import FileIndexDB
            prior_models = await _run_db(FileIndexDB.get_indexed_models_for_folder, folder_id)
            stale_models = [m for m in prior_models if m != ctx.model_name]
            if stale_models:
                raise ValueError(
                    f"Folder {folder_id} has indexes built with embedding model "
                    f"{stale_models} but current config uses '{ctx.model_name}'. "
                    f"Mixing embedding spaces silently corrupts retrieval — "
                    f"reindex the folder (POST /api/rag/files/{folder_id}/reindex) "
                    f"to switch models."
                )

            log_op(
                logger, "INDEX_START",
                folder_id=folder_id, file_id=file_id, token=token,
                msg=(
                    f"table={vector_store_table} "
                    f"leaf_size={ctx.leaf_chunk_size} "
                    f"parent_target={ctx.parent_target_tokens}"
                ),
            )

            # Docling parsing + RapidOCR is synchronous CPU-bound work that would stall the
            # event loop (~32s for an 11MB image-heavy PDF), so offload it to a thread pool
            # to keep serving upload/query requests during indexing.
            logger.debug(
                f"[INDEX_LOAD] fid={folder_id} file={file_id} | "
                f"path={file_record.file_path} name={file_record.file_name}"
            )
            # Report the loading stage first: docling/OCR is the biggest chunk-progress blind
            # spot, so the job status must not sit with no stage during parsing.
            if _progress_cb:
                try:
                    _progress_cb("loading", None, None)
                except Exception:
                    pass  # a progress-report failure must not affect indexing
            load_t0 = time.perf_counter()
            document = await asyncio.to_thread(
                ctx.indexer.load_document_from_file,
                file_path=file_record.file_path,
                file_id=str(file_id),
                file_name=file_record.file_name,
                # Page-level parse progress (PDF/OCR); the keepalive-wrapped cb also refreshes the heartbeat.
                progress_cb=_progress_cb,
                metadata={
                    "folder_id": folder_id,
                    "folder_name": folder.name,
                    "mime_type": file_record.mime_type,
                    "file_size": file_record.file_size,
                },
            )
            if _timings_out is not None:
                _timings_out["load_ms"] = round((time.perf_counter() - load_t0) * 1000, 1)
            logger.debug(
                f"[INDEX_LOAD] fid={folder_id} file={file_id} | "
                f"text_len={len(document.text) if document.text else 0}"
            )

            # Gate 2: re-check after Docling load, before the expensive context-gen + embedding stages.
            if not await _run_db(self._file_still_exists, file_id):
                log_warn(
                    logger, "INDEX_ABORT",
                    folder_id=folder_id, file_id=file_id, token=token,
                    msg=f"file {file_id} was deleted during indexing — aborting (gate 2)",
                )
                raise RAGFileNotFoundError(file_id=str(file_id))

            # Index hierarchically; returns {"leaves": N, "parents": M}.
            idx_t0 = time.perf_counter()
            try:
                index_stats = await ctx.indexer.index_document(
                    document=document,
                    vector_store=vector_store,
                    progress_cb=_progress_cb,
                    should_abort=self._make_should_abort(file_id),
                )
            except IndexingAbortedError as abort_exc:
                # An in-stage checkpoint found the file deleted mid-way: same abort path as the gates.
                log_warn(
                    logger, "INDEX_ABORT",
                    folder_id=folder_id, file_id=file_id, token=token,
                    msg=f"{abort_exc} — aborting (mid-stage probe)",
                )
                raise RAGFileNotFoundError(file_id=str(file_id))
            if _timings_out is not None:
                _timings_out["index_ms"] = round((time.perf_counter() - idx_t0) * 1000, 1)
            num_leaves = index_stats.get("leaves", 0)
            num_parents = index_stats.get("parents", 0)

            log_op(
                logger, "INDEX_DONE",
                folder_id=folder_id, file_id=file_id, token=token,
                msg=f"{num_leaves} leaves + {num_parents} parents",
            )

            # Gate 3: last chance. If the file is gone now, the FileIndex FK insert below
            # would fail anyway, so abort cleanly.
            if not await _run_db(self._file_still_exists, file_id):
                # Vectors are already written to the table at this point (index_document
                # includes the add). Aborting without cleanup would leave orphan chunks
                # forever (no FileIndex, no File row) that keep surfacing in retrieval, so
                # purge the freshly-written chunks first.
                try:
                    from src.domain.rag.vector_store_manager import VectorStoreManager
                    removed = await _run_db(
                        VectorStoreManager.delete_chunks_by_file,
                        vector_store_table, str(file_id),
                    )
                    log_warn(
                        logger, "INDEX_ABORT_PURGE",
                        folder_id=folder_id, file_id=file_id, token=token,
                        msg=f"purged {removed} freshly-written chunk(s) of deleted file",
                    )
                except Exception as purge_err:
                    log_err(logger, "INDEX_ABORT_PURGE_FAIL", purge_err,
                            folder_id=folder_id, file_id=file_id, token=token)
                log_warn(
                    logger, "INDEX_ABORT",
                    folder_id=folder_id, file_id=file_id, token=token,
                    msg=f"file {file_id} was deleted during indexing — aborting (gate 3)",
                )
                raise RAGFileNotFoundError(file_id=str(file_id))

            # Record index metadata: chunk_size is the leaf size, num_chunks the leaf count
            # (the unit actually retrieved).
            index_record = await _run_db(
                ctx.indexing_service.record_index_success,
                file_id=file_id,
                folder_id=folder_id,
                index_id=f"file_{file_id}",
                vector_store_table=vector_store_table,
                chunk_size=ctx.leaf_chunk_size,
                chunk_overlap=ctx.chunk_overlap,
                embedding_model=ctx.model_name,
                num_chunks=num_leaves,
                content_hash=getattr(file_record, "content_hash", None),
            )

            await _run_db(ctx.indexing_service.invalidate_folder_caches, folder_id)
            ctx.invalidate_query_engine_cache(folder_id)

            return IndexDocumentResponse(
                file_id=str(file_id),
                index_id=index_record.index_id,
                num_chunks=num_leaves,
                status="indexed",
                indexed_at=index_record.indexed_at.isoformat(),
                message=f"Indexed hierarchically: {num_leaves} leaves + {num_parents} parents",
            )

        except DomainException:
            raise
        except Exception as e:
            log_err(
                logger, "INDEX_FAIL", e,
                folder_id=folder_id, file_id=file_id, token=token,
            )
            try:
                # Pass the real folder_id/model/size so the failed row reflects the actual config, not column defaults.
                ctx.indexing_service.mark_index_failed(
                    file_id,
                    str(e),
                    folder_id=folder_id,
                    embedding_model=ctx.model_name,
                    chunk_size=ctx.leaf_chunk_size,
                    chunk_overlap=ctx.chunk_overlap,
                )
            except Exception:
                pass
            raise FileIndexingError(file_id=str(file_id), reason=str(e))

    async def index_folder(
        self,
        folder_id: int,
        token: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        skip_existing: bool = True,
        progress_tracker: Optional["IndexProgressTracker"] = None,
    ) -> IndexFolderResponse:
        """Index all documents in a folder into the vector store."""
        ctx = self._ctx

        try:
            # Validate folder access upfront
            await _run_db(ctx.indexing_service.get_folder, folder_id, token)

            files = await _run_db(ctx.indexing_service.list_files, folder_id)
            total_files = len(files)

            if progress_tracker:
                progress_tracker.on_started(total_files, skip_existing)

            if not files:
                log_op(
                    logger, "INDEX_FOLDER",
                    folder_id=folder_id, token=token, msg="no files found",
                )
                response = IndexFolderResponse(
                    folder_id=folder_id, total_files=0, successful=0,
                    failed=0, skipped=0, results=[],
                    message="No files found in folder",
                )
                if progress_tracker:
                    progress_tracker.mark_completed(response.model_dump())
                return response

            log_op(
                logger, "INDEX_FOLDER_START",
                folder_id=folder_id, token=token,
                msg=f"{len(files)} files, skip_existing={skip_existing}",
            )

            successful = 0
            failed = 0
            skipped = 0
            results: List[FileIndexResult] = []
            processed_count = 0
            per_file_timeout = _resolve_per_file_timeout()  # bounds each file

            for file_record in files:
                file_id = file_record.id
                filename = file_record.file_name

                if progress_tracker:
                    progress_tracker.on_file_started(
                        current_index=processed_count + 1,
                        file_id=str(file_id),
                        file_name=filename,
                    )

                stage_timings: Dict[str, float] = {}
                file_t0 = time.perf_counter()
                try:
                    if skip_existing:
                        existing_index = await _run_db(ctx.indexing_service.get_index_for_file, file_id)
                        # Only status=indexed counts as already indexed; failed/deleted rows
                        # must not block a retry, or a failed file could never be recovered via /index.
                        if existing_index and getattr(existing_index, "status", None) == "indexed":
                            log_warn(
                                logger, "INDEX_SKIP",
                                folder_id=folder_id, file_id=file_id, token=token,
                                msg=f"already indexed: {filename}",
                            )
                            skipped += 1
                            processed_count += 1
                            results.append(FileIndexResult(
                                file_id=str(file_id), filename=filename,
                                status="skipped",
                                message="File is already indexed",
                                num_chunks=None,
                            ))
                            if progress_tracker:
                                progress_tracker.on_file_completed(
                                    processed_files=processed_count,
                                    status="skipped",
                                    message="File is already indexed",
                                )
                                progress_tracker.record_file_timing(_make_timing_entry(
                                    file_id, filename, "skipped",
                                    file_t0, stage_timings,
                                ))
                            continue

                    result = await _with_timeout(
                        self.index_document(
                            file_record=file_record,
                            token=token,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            folder_id_override=folder_id,
                            _timings_out=stage_timings,
                            _progress_cb=(
                                progress_tracker.on_chunk_progress
                                if progress_tracker else None
                            ),
                        ),
                        per_file_timeout,
                        filename,
                    )

                    successful += 1
                    processed_count += 1
                    results.append(FileIndexResult(
                        file_id=str(file_id), filename=filename,
                        status="success",
                        message=result.message,
                        num_chunks=result.num_chunks,
                    ))
                    log_op(
                        logger, "INDEX_FILE_DONE",
                        folder_id=folder_id, file_id=file_id, token=token,
                        msg=f"{result.num_chunks} chunks | {filename}",
                    )
                    if progress_tracker:
                        progress_tracker.on_file_completed(
                            processed_files=processed_count,
                            status="success",
                            message=result.message,
                        )
                        progress_tracker.record_file_timing(_make_timing_entry(
                            file_id, filename, "success", file_t0, stage_timings,
                            chunks=result.num_chunks,
                        ))

                except Exception as e:
                    failed += 1
                    error_message = str(e)
                    processed_count += 1
                    status_label = "timeout" if isinstance(e, TimeoutError) else "failed"
                    results.append(FileIndexResult(
                        file_id=str(file_id), filename=filename,
                        status="failed",
                        message=error_message,
                        num_chunks=None,
                    ))
                    log_err(
                        logger, "INDEX_FILE_FAIL", e,
                        folder_id=folder_id, file_id=file_id, token=token,
                    )
                    # Backstop write of the failed record. A timeout cancels via
                    # asyncio.wait_for with a CancelledError (a BaseException), which
                    # index_document's inner except Exception cannot catch, so its own
                    # mark_index_failed never runs and the timed-out file would get no
                    # record and no frontend tag. For non-timeout failures index_document
                    # already wrote it, and re-upserting the same row is harmless.
                    try:
                        ctx.indexing_service.mark_index_failed(
                            file_id,
                            error_message,
                            folder_id=folder_id,
                            embedding_model=ctx.model_name,
                            chunk_size=ctx.leaf_chunk_size,
                            chunk_overlap=ctx.chunk_overlap,
                        )
                    except Exception as _mf_err:
                        # Don't swallow a failed backstop write: a DB disconnect/conflict
                        # would leave the frontend without a tag and no trace.
                        logger.warning(
                            f"mark_index_failed fallback also failed for "
                            f"file {file_id}: {_mf_err}"
                        )
                    if progress_tracker:
                        progress_tracker.on_file_completed(
                            processed_files=processed_count,
                            status="failed",
                            message=error_message,
                        )
                        progress_tracker.record_file_timing(_make_timing_entry(
                            file_id, filename, status_label, file_t0, stage_timings,
                        ))

                except asyncio.CancelledError:
                    # A job-level cancel/timeout injects CancelledError (a BaseException the
                    # except Exception above can't catch); the in-flight file would be neither
                    # indexed nor failed and vanish from the frontend. Write a failed record,
                    # then re-raise. mark_index_failed is synchronous, so it is safe during cancel.
                    try:
                        ctx.indexing_service.mark_index_failed(
                            file_id,
                            "aborted: job cancelled or job-level timeout",
                            folder_id=folder_id,
                            embedding_model=ctx.model_name,
                            chunk_size=ctx.leaf_chunk_size,
                            chunk_overlap=ctx.chunk_overlap,
                        )
                    except Exception as _mf_err:
                        logger.warning(
                            f"mark_index_failed on cancel failed for file {file_id}: {_mf_err}"
                        )
                    raise

            summary_message = (
                f"Folder indexing completed: {successful} successful, "
                f"{failed} failed, {skipped} skipped"
            )
            log_op(
                logger, "INDEX_FOLDER_DONE",
                folder_id=folder_id, token=token, msg=summary_message,
            )

            await _run_db(ctx.indexing_service.invalidate_folder_caches, folder_id)

            response = IndexFolderResponse(
                folder_id=folder_id,
                total_files=total_files,
                successful=successful,
                failed=failed,
                skipped=skipped,
                results=results,
                message=summary_message,
            )
            if progress_tracker:
                progress_tracker.mark_completed(response.model_dump())
            return response

        except DomainException as domain_exc:
            if progress_tracker:
                progress_tracker.mark_failed(str(domain_exc))
            raise
        except Exception as e:
            log_err(logger, "INDEX_FOLDER_FAIL", e, folder_id=folder_id, token=token)
            if progress_tracker:
                progress_tracker.mark_failed(str(e))
            raise ValueError(f"Failed to index folder: {str(e)}")

    async def index_files(
        self,
        file_ids: List[str],
        folder_id: int,
        token: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        progress_tracker: Optional["IndexProgressTracker"] = None,
    ) -> IndexFolderResponse:
        """Index a specified subset of file_ids (for auto-index after upload).

        Like index_folder but only processes the files in file_ids, and does not skip
        already-indexed files, because the caller wants to redo or complete the files
        just uploaded. Returns an IndexFolderResponse with per-file detail and stats.
        """
        ctx = self._ctx

        try:
            await _run_db(ctx.indexing_service.get_folder, folder_id, token)

            files = await _run_db(ctx.indexing_service.list_files, folder_id)
            files_to_index = [f for f in files if str(f.id) in file_ids]
            total_files = len(files_to_index)

            if progress_tracker:
                progress_tracker.on_started(total_files, skip_existing=False)

            if not files_to_index:
                log_op(
                    logger, "INDEX_FILES",
                    folder_id=folder_id, token=token,
                    msg="no matching files found",
                )
                response = IndexFolderResponse(
                    folder_id=folder_id, total_files=0, successful=0,
                    failed=0, skipped=0, results=[],
                    message="No files found to index",
                )
                if progress_tracker:
                    progress_tracker.mark_completed(response.model_dump())
                return response

            log_op(
                logger, "INDEX_FILES_START",
                folder_id=folder_id, token=token,
                msg=f"{len(files_to_index)} files",
            )

            successful = 0
            failed = 0
            skipped = 0
            results: List[FileIndexResult] = []
            processed_count = 0
            per_file_timeout = _resolve_per_file_timeout()  # bounds each file

            for file_record in files_to_index:
                file_id = file_record.id
                filename = file_record.file_name

                if progress_tracker:
                    progress_tracker.on_file_started(
                        current_index=processed_count + 1,
                        file_id=str(file_id),
                        file_name=filename,
                    )

                stage_timings: Dict[str, float] = {}
                file_t0 = time.perf_counter()
                try:
                    result = await _with_timeout(
                        self.index_document(
                            file_record=file_record,
                            token=token,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            folder_id_override=folder_id,
                            _timings_out=stage_timings,
                            _progress_cb=(
                                progress_tracker.on_chunk_progress
                                if progress_tracker else None
                            ),
                        ),
                        per_file_timeout,
                        filename,
                    )
                    successful += 1
                    processed_count += 1
                    results.append(FileIndexResult(
                        file_id=str(file_id), filename=filename,
                        status="success",
                        message=result.message,
                        num_chunks=result.num_chunks,
                    ))
                    log_op(
                        logger, "INDEX_FILE_DONE",
                        folder_id=folder_id, file_id=file_id, token=token,
                        msg=f"{result.num_chunks} chunks | {filename}",
                    )
                    if progress_tracker:
                        progress_tracker.on_file_completed(
                            processed_files=processed_count,
                            status="success",
                            message=result.message,
                        )
                        progress_tracker.record_file_timing(_make_timing_entry(
                            file_id, filename, "success", file_t0, stage_timings,
                            chunks=result.num_chunks,
                        ))

                except Exception as e:
                    failed += 1
                    error_message = str(e)
                    processed_count += 1
                    status_label = "timeout" if isinstance(e, TimeoutError) else "failed"
                    results.append(FileIndexResult(
                        file_id=str(file_id), filename=filename,
                        status="failed",
                        message=error_message,
                        num_chunks=None,
                    ))
                    log_err(
                        logger, "INDEX_FILE_FAIL", e,
                        folder_id=folder_id, file_id=file_id, token=token,
                    )
                    # Backstop write of the failed record. A timeout cancels via
                    # asyncio.wait_for with a CancelledError (a BaseException), which
                    # index_document's inner except Exception cannot catch, so its own
                    # mark_index_failed never runs and the timed-out file would get no
                    # record and no frontend tag. For non-timeout failures index_document
                    # already wrote it, and re-upserting the same row is harmless.
                    try:
                        ctx.indexing_service.mark_index_failed(
                            file_id,
                            error_message,
                            folder_id=folder_id,
                            embedding_model=ctx.model_name,
                            chunk_size=ctx.leaf_chunk_size,
                            chunk_overlap=ctx.chunk_overlap,
                        )
                    except Exception as _mf_err:
                        # Don't swallow a failed backstop write: a DB disconnect/conflict
                        # would leave the frontend without a tag and no trace.
                        logger.warning(
                            f"mark_index_failed fallback also failed for "
                            f"file {file_id}: {_mf_err}"
                        )
                    if progress_tracker:
                        progress_tracker.on_file_completed(
                            processed_files=processed_count,
                            status="failed",
                            message=error_message,
                        )
                        progress_tracker.record_file_timing(_make_timing_entry(
                            file_id, filename, status_label, file_t0, stage_timings,
                        ))

                except asyncio.CancelledError:
                    # A job-level cancel/timeout injects CancelledError (a BaseException the
                    # except Exception above can't catch); the in-flight file would be neither
                    # indexed nor failed and vanish from the frontend. Write a failed record,
                    # then re-raise. mark_index_failed is synchronous, so it is safe during cancel.
                    try:
                        ctx.indexing_service.mark_index_failed(
                            file_id,
                            "aborted: job cancelled or job-level timeout",
                            folder_id=folder_id,
                            embedding_model=ctx.model_name,
                            chunk_size=ctx.leaf_chunk_size,
                            chunk_overlap=ctx.chunk_overlap,
                        )
                    except Exception as _mf_err:
                        logger.warning(
                            f"mark_index_failed on cancel failed for file {file_id}: {_mf_err}"
                        )
                    raise

            summary_message = (
                f"File indexing completed: {successful} successful, {failed} failed"
            )
            log_op(
                logger, "INDEX_FILES_DONE",
                folder_id=folder_id, token=token, msg=summary_message,
            )

            await _run_db(ctx.indexing_service.invalidate_folder_caches, folder_id)

            response = IndexFolderResponse(
                folder_id=folder_id,
                total_files=total_files,
                successful=successful,
                failed=failed,
                skipped=skipped,
                results=results,
                message=summary_message,
            )
            if progress_tracker:
                progress_tracker.mark_completed(response.model_dump())
            return response

        except DomainException as domain_exc:
            if progress_tracker:
                progress_tracker.mark_failed(str(domain_exc))
            raise
        except Exception as e:
            error_msg = f"Failed to index files: {str(e)}"
            logger.error(error_msg)
            if progress_tracker:
                progress_tracker.mark_failed(error_msg)
            raise ValueError(error_msg)

    async def trigger_auto_index(
        self,
        file_ids: List[str],
        folder_id: int,
        token: str,
        auto_index: bool = True,
    ) -> Dict[str, Any]:
        """Trigger the background auto-index job after a successful upload.

        auto_index=False or empty file_ids returns immediately without starting. Returns
        an IndexingMetadata-shaped dict {auto_index_enabled, job_id?, status?, total_files?, message}.
        """
        try:
            from src.domain.rag.index_job_manager import IndexingJobManager

            if not auto_index or not file_ids:
                return {
                    "auto_index_enabled": False,
                    "message": (
                        "Auto-indexing disabled or no files to index. "
                        "Call /api/rag/files/{folder_id}/index to index manually"
                    ),
                }

            chunk_size = self._ctx.leaf_chunk_size
            chunk_overlap = self._ctx.chunk_overlap

            job_manager = IndexingJobManager.get_instance()
            # This service exposes index_files directly, so it can act as the adapter the
            # job manager calls back into.
            job_state = await job_manager.start_indexing_files(
                self,
                file_ids=file_ids,
                folder_id=folder_id,
                token=token,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            log_op(
                logger, "AUTO_INDEX",
                folder_id=folder_id, token=token,
                msg=f"{len(file_ids)} files, job_id={job_state['job_id']}",
            )

            return {
                "auto_index_enabled": True,
                "job_id": job_state["job_id"],
                "status": job_state["status"],
                "total_files": len(file_ids),
                "message": f"Indexing started in background for {len(file_ids)} file(s)",
            }

        except Exception as e:
            log_err(logger, "AUTO_INDEX_FAIL", e, folder_id=folder_id, token=token)
            return {
                "auto_index_enabled": True,
                "message": (
                    f"Warning: Auto-indexing failed to start: {str(e)}. "
                    f"File uploaded successfully. You can index manually later."
                ),
            }
