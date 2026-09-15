
"""RAG Indexing REST API

Handles indexing-related operations only:
- POST   /api/rag/files/{folder_id}/index                       index an entire folder (background, async)
- POST   /api/rag/file/{file_id}/index                          index a single file
- POST   /api/rag/files/{folder_id}/reindex                     reindex an entire folder
- GET    /api/rag/indexed-files                                 list indexed files
- GET    /api/rag/files/{folder_id}/index/jobs/{job_id}         query background job status
- GET    /api/rag/files/{folder_id}/index/jobs/{job_id}/events  SSE realtime progress stream
- DELETE /api/rag/files/{folder_id}/index/jobs/{job_id}         cancel a background indexing job
- GET    /api/rag/file/{file_id}/index/status                   query single-file index status
- DELETE /api/rag/files/{folder_id}/index                       delete an entire folder's index
- DELETE /api/rag/file/{file_id}/index                          delete a single file's index

Query endpoints have moved to rag_query.py.
"""

import asyncio
import json
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from db.cached_folderdb import CachedFolderDB
from db.db import File, Folder
from src.adapter.folder import FolderAdapter
from src.adapter.rag import get_rag_adapter
from src.api.dependencies.auth import (
    extract_token,
    get_file_by_id,
    get_folder_by_id,
    get_folder_by_name,
)
from src.api.router.response import (
    ErrorDetailResponse,
    FileIndexStatusEnvelope,
    FileIndexStatusResponse,
    FileRequest,
    IndexDocumentEnvelope,
    IndexedFilesResponse,
    IndexJobCancelResponse,
    IndexJobEnvelope,
    IndexJobListResponse,
    IndexJobStatusResponse,
    IndexRequest,
)
from src.auth.dependencies import authenticate_request
from src.domain.exceptions import ConflictError, FolderNotFoundError
from src.domain.rag.index_job_manager import IndexingJobManager, JobStatus
from src.log import get_api_logger, log_err, log_op, log_warn

logger = get_api_logger()

# Router-level tag left empty; each endpoint declares its own group (Indexing/Jobs/Status/Cleanup)
router = APIRouter(dependencies=[Depends(authenticate_request)])

# Tag constants — keep in sync with openapi_tags in app.py (the order is defined there too)
_TAG_INDEXING = "RAG: Indexing"
_TAG_JOBS = "RAG: Jobs"
_TAG_STATUS = "RAG: Status"
_TAG_CLEANUP = "RAG: Cleanup"

job_manager = IndexingJobManager.get_instance()


# Index operations

_INDEX_REQUEST_EXAMPLES = {
    "use_config_defaults": {
        "summary": "Use config defaults (most common)",
        "description": "Send an empty body; indexing reads defaults from `config.rag.chunking`.",
        "value": {},
    },
    "explicit_match_config": {
        "summary": "Explicitly match current config values",
        "description": "Copy the config defaults into the request to tune just one of them.",
        "value": {"chunk_size": 256, "chunk_overlap": 50},
    },
    "smaller_chunks": {
        "summary": "Smaller chunks (higher precision)",
        "description": "Smaller leaf chunks improve retrieval precision but produce more chunks.",
        "value": {"chunk_size": 128, "chunk_overlap": 30},
    },
}


# Resource = IndexJob; Response = {data: IndexJobStatusResponse, message} (background job; returns job_id immediately)
@router.post(
    "/files/{folder_id}/index",
    tags=[_TAG_INDEXING],
    response_model=IndexJobEnvelope,
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder does not exist"},
        409: {"model": ErrorDetailResponse, "description": "This folder already has an in-flight indexing job (folder lock)"},
        500: {"model": ErrorDetailResponse},
    },
)
async def index_folder_endpoint(
    folder_id: int,
    index_request: IndexRequest = Body(..., openapi_examples=_INDEX_REQUEST_EXAMPLES),
    skip_existing: bool = True,
    token: str = Depends(extract_token),
    folder: Folder = Depends(get_folder_by_id),
):
    """Index all files in an entire folder in the background."""
    try:
        log_op(logger, "API_INDEX_FOLDER", folder_id=folder_id, token=token,
               msg=f"skip_existing={skip_existing}")

        folder_list = await FolderAdapter.get_folder(id=folder_id, user_token=token)
        if not folder_list:
            raise HTTPException(status_code=404, detail=f"folder {folder_id} not found")

        job_state = await job_manager.start_indexing(
            get_rag_adapter(),
            folder_id=folder_id,
            token=token,
            chunk_size=index_request.chunk_size,
            chunk_overlap=index_request.chunk_overlap,
            skip_existing=skip_existing,
        )

        job_response = IndexJobStatusResponse.model_validate(job_state)
        return {"data": job_response.model_dump(), "message": "Indexing job started"}

    except FolderNotFoundError as e:
        log_warn(logger, "API_INDEX_FOLDER_404", folder_id=folder_id, token=token, msg=str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except ConflictError as e:
        log_warn(logger, "API_INDEX_FOLDER_409", folder_id=folder_id, token=token, msg=e.message)
        raise HTTPException(status_code=409, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        log_err(logger, "API_INDEX_FOLDER_FAIL", e, folder_id=folder_id, token=token)
        raise HTTPException(status_code=500, detail=f"Failed to start indexing: {str(e)}")


# Resource = FileIndex; Response = {data: IndexDocumentResponse, message} (synchronous; returns only after indexing actually completes)
@router.post(
    "/file/{file_id}/index",
    tags=[_TAG_INDEXING],
    response_model=IndexDocumentEnvelope,
    responses={
        404: {"model": ErrorDetailResponse, "description": "file does not exist or no permission"},
        500: {"model": ErrorDetailResponse, "description": "indexing failed (Docling / embedding, etc.)"},
    },
)
async def index_single_file_endpoint(
    file_id: UUID,
    index_request: IndexRequest = Body(..., openapi_examples=_INDEX_REQUEST_EXAMPLES),
    force: bool = False,
    token: str = Depends(extract_token),
    file: File = Depends(get_file_by_id),
):
    """Synchronously index a single file.

    force=true skips the content_hash idempotency short-circuit and forces the whole pipeline to
    rerun (single-file rebuild after a config change; the file's old vectors are cleared before
    writing, so no duplicates remain).
    """
    try:
        log_op(logger, "API_INDEX_FILE", file_id=file_id, token=token,
               msg=f"chunk_size={index_request.chunk_size} force={force}")

        file_request = FileRequest(
            id=file.id,
            file_name=file.file_name,
            file_path=file.file_path,
            folder_id=file.folder_id,
            mime_type=file.mime_type,
            file_size=file.file_size,
            content_hash=getattr(file, "content_hash", None),  # D3 idempotency
        )

        result = await get_rag_adapter().index_document(
            file_record=file_request,
            token=token,
            chunk_size=index_request.chunk_size,
            chunk_overlap=index_request.chunk_overlap,
            force=force,
        )

        return {
            "data": result.model_dump(),
            "message": f"File indexed successfully with {result.num_chunks} chunks",
        }
    except HTTPException:
        raise
    except Exception as e:
        log_err(logger, "API_INDEX_FILE_FAIL", e, file_id=file_id, token=token)
        raise HTTPException(status_code=500, detail=f"Failed to index file: {str(e)}")


# Resource = IndexJob; Response = {data: IndexJobStatusResponse, message} (deletes the folder index first, then starts a new job)
@router.post(
    "/files/{folder_id}/reindex",
    tags=[_TAG_INDEXING],
    response_model=IndexJobEnvelope,
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder does not exist"},
        409: {"model": ErrorDetailResponse, "description": "This folder already has an in-flight job"},
        500: {"model": ErrorDetailResponse},
    },
)
async def reindex_folder_endpoint(
    folder_id: int,
    reindex_request: IndexRequest = Body(..., openapi_examples=_INDEX_REQUEST_EXAMPLES),
    skip_existing: bool = False,
    token: str = Depends(extract_token),
    folder: Folder = Depends(get_folder_by_id),
):
    """Delete the old index first, then reindex the entire folder in the background."""
    try:
        logger.info(
            f"API: Reindexing folder_id={folder_id} chunk_size={reindex_request.chunk_size} "
            f"chunk_overlap={reindex_request.chunk_overlap}"
        )

        # Failing to delete the old index must abort: proceeding after only a warning would start
        # the reindex with the index uncleared, and the content_hash short-circuit would make it a
        # silent no-op.
        try:
            delete_result = await get_rag_adapter().delete_folder_index(
                folder_id=folder_id, token=token
            )
            logger.info(f"Deleted {delete_result.successful} file indices before reindexing")
        except Exception as e:
            log_err(logger, "API_REINDEX_DELETE_FAIL", e, folder_id=folder_id)
            raise HTTPException(
                status_code=500,
                detail=f"Reindex aborted: failed to clear existing index ({e}). "
                       "Nothing was rebuilt — retry when the cause is resolved.",
            )

        # The folder-lock gap between delete and start could be grabbed by auto-index
        # (start_indexing has reject semantics). The index is now cleared, so giving up with a 409
        # would leave an empty index — retry acquiring the lock a few times, returning 409 only if
        # it can't be obtained (with a message clearly telling the user to retry).
        job_state = None
        for attempt in range(3):
            try:
                job_state = await job_manager.start_indexing(
                    get_rag_adapter(),
                    folder_id=folder_id,
                    token=token,
                    chunk_size=reindex_request.chunk_size,
                    chunk_overlap=reindex_request.chunk_overlap,
                    skip_existing=skip_existing,
                )
                break
            except ConflictError:
                if attempt == 2:
                    raise HTTPException(
                        status_code=409,
                        detail="Index cleared but folder busy (another job grabbed the lock). "
                               "Existing index has been deleted — start indexing again once the folder is free.",
                    )
                await asyncio.sleep(1.5)

        job_response = IndexJobStatusResponse.model_validate(job_state)
        return {"data": job_response.model_dump(), "message": "Reindexing job started"}

    except ConflictError as e:
        log_warn(logger, "API_REINDEX_409", folder_id=folder_id, token=token, msg=e.message)
        raise HTTPException(status_code=409, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        log_err(logger, "API_REINDEX_FAIL", e, folder_id=folder_id, token=token)
        raise HTTPException(status_code=500, detail=f"Failed to reindex folder: {str(e)}")


# Read-only status

# Resource = List[FileIndex]; Response = {data: [IndexedFileEntry...], message}
@router.get(
    "/indexed-files",
    tags=[_TAG_STATUS],
    response_model=IndexedFilesResponse,
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder does not exist"},
        500: {"model": ErrorDetailResponse},
    },
)
async def get_indexed_files(
    folder_id: int,
    token: str = Depends(extract_token),
    folder: Folder = Depends(get_folder_by_id),
):
    """List all indexed files in a specified folder."""
    try:
        logger.info(f"API: Getting indexed files for folder_id={folder_id}")
        result = await get_rag_adapter().get_indexed_files(folder_id=folder_id)
        return {"data": result, "message": f"Found {len(result)} indexed files"}
    except HTTPException:
        raise
    except Exception as e:
        log_err(logger, "API_GET_INDEXED_FAIL", e, folder_id=folder_id, token=token)
        raise HTTPException(status_code=500, detail=f"Failed to get indexed files: {str(e)}")


# Resource = IndexJob;Response = {data: IndexJobStatusResponse, message}
@router.get(
    "/files/{folder_id}/index/jobs/{job_id}",
    tags=[_TAG_JOBS],
    response_model=IndexJobEnvelope,
    responses={
        404: {"model": ErrorDetailResponse, "description": "job_id does not exist or does not belong to this folder"},
        500: {"model": ErrorDetailResponse},
    },
)
async def get_index_job_status(
    folder_id: int,
    job_id: str,
    token: str = Depends(extract_token),
    folder: Folder = Depends(get_folder_by_id),
):
    """Query the status of a background indexing job."""
    try:
        status = await job_manager.get_status(job_id)

        if not status or status.get("folder_id") != folder_id:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        job_response = IndexJobStatusResponse.model_validate(status)
        return {"data": job_response.model_dump(), "message": "Indexing job status retrieved"}

    except HTTPException:
        raise
    except Exception as e:
        log_err(logger, "API_JOB_STATUS_FAIL", e, folder_id=folder_id)
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


# Response = {counts: {status: count}, jobs: [IndexJobStatusResponse...]}
@router.get(
    "/index/jobs",
    tags=[_TAG_JOBS],
    response_model=IndexJobListResponse,
    responses={
        400: {"model": ErrorDetailResponse, "description": "status query param is not a valid value"},
        500: {"model": ErrorDetailResponse},
    },
)
async def list_index_jobs(
    status: Optional[str] = None,
    folder_id: Optional[int] = None,
    token: str = Depends(extract_token),
):
    """List indexing jobs + per-status counts (returns only jobs for folders owned by the token).

    Query params:
      - status: pending / running / succeeded / partial_success / failed / cancelled
                (optional; returns only that status)
      - folder_id: return only that folder's jobs (optional; 404 if it doesn't belong to this token)

    Returns:
      {
        "counts": {"pending": N, "running": N, "succeeded": N, "partial_success": N,
                   "failed": N, "cancelled": N, "total": N},
        "jobs": [ {job_id, folder_id, status, progress, ...}, ... ]
      }
    """
    try:
        status_filter = None
        if status:
            try:
                status_filter = JobStatus(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status '{status}'. Use one of: "
                           f"{[s.value for s in JobStatus]}",
                )

        # Ownership filter — without it, any valid token could see site-wide jobs (filenames/errors/timings)
        if folder_id is not None:
            folder = CachedFolderDB.get_by_id(folder_id)
            if not folder or folder.user_token != token:
                raise HTTPException(status_code=404, detail=f"Folder not found: {folder_id}")
            scoped = job_manager.list_jobs(folder_id=folder_id)
        else:
            # use_cache=False: folder creation goes through FolderDB.create, which doesn't
            # invalidate the folders:user:{token} cache (TTL 180s) — using the cache would hide a
            # newly created folder's jobs here for up to 3 minutes. Query the DB directly (there's a
            # user_token index, so it's cheap).
            allowed_ids = {
                f.id for f in CachedFolderDB.get_by_user_token(token, use_cache=False)
            }
            scoped = [
                j for j in job_manager.list_jobs()
                if j.get("folder_id") in allowed_ids
            ]

        # Counts must also be scoped to the caller's own jobs (preserving original semantics: unaffected by the status filter)
        counts = {s.value: 0 for s in JobStatus}
        for j in scoped:
            counts[j["status"]] = counts.get(j["status"], 0) + 1
        counts["total"] = len(scoped)

        jobs = (
            [j for j in scoped if j["status"] == status_filter.value]
            if status_filter is not None else scoped
        )
        return {"counts": counts, "jobs": jobs}
    except HTTPException:
        raise
    except Exception as e:
        log_err(logger, "API_JOBS_LIST_FAIL", e)
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


# Statuses that mean "nothing more will happen on this job"
_TERMINAL_JOB_STATUSES = {
    "succeeded", "partial_success", "failed", "cancelled",
}


# Stream; non-JSON — each event is a serialized IndexJobStatusResponse. Content-Type: text/event-stream
@router.get(
    "/files/{folder_id}/index/jobs/{job_id}/events",
    tags=[_TAG_JOBS],
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Server-Sent Events stream; each `data: {...}` is a complete IndexJobStatusResponse JSON, closing automatically at terminal status",
        },
        404: {"model": ErrorDetailResponse, "description": "job_id does not exist"},
    },
)
async def stream_index_job_events(
    folder_id: int,
    job_id: str,
    request: Request,
    token: str = Depends(extract_token),
    folder: Folder = Depends(get_folder_by_id),
):
    """SSE stream: push job status changes (replacing polling).

    Every status / progress / per-file completion / timing write pushes a `data: {full state JSON}`
    event. The stream closes automatically once the job reaches a terminal status. A `:ping` is sent
    after 15s idle to prevent proxy disconnects.

    Args:
        folder_id: The folder the job belongs to.
        job_id: The job id to subscribe to.
        request: FastAPI Request (used to detect client disconnect).
        token: User token (permission check).
        folder: Injected by the get_folder_by_id dependency.

    Returns:
        StreamingResponse, Content-Type: text/event-stream.
    """
    # Subscribe before reading the snapshot. The reverse order leaves a gap: if the job goes
    # terminal after the snapshot is read but before subscribing, the terminal event is delivered to
    # a not-yet-existing subscriber and lost -> the snapshot shows a non-terminal status and the
    # stream afterward only pings every 15s, hanging forever on an already-finished job. Subscribing
    # first misses nothing.
    queue = job_manager.subscribe(job_id)
    initial = await job_manager.get_status(job_id)
    if not initial or initial.get("folder_id") != folder_id:
        job_manager.unsubscribe(job_id, queue)
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    async def event_stream():
        try:
            # 1. Send the current snapshot first, so the client renders the correct state on connect without polling
            yield f"data: {json.dumps(initial)}\n\n"
            if initial.get("status") in _TERMINAL_JOB_STATUSES:
                return

            # 2. Consume all updates after subscribe (which preceded the snapshot); terminal isn't missed
            while True:
                # Stop if the client disconnects; otherwise we'd emit to a dead socket indefinitely
                if await request.is_disconnected():
                    return
                try:
                    state = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(state)}\n\n"
                if state.get("status") in _TERMINAL_JOB_STATUSES:
                    return
        finally:
            job_manager.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # Disable common buffering interceptors so events flush promptly.
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# Response = {data: {cancelled: bool, job_id: str}, message}
@router.delete(
    "/files/{folder_id}/index/jobs/{job_id}",
    tags=[_TAG_JOBS],
    response_model=IndexJobCancelResponse,
    responses={
        404: {"model": ErrorDetailResponse, "description": "job_id does not exist or does not belong to this folder"},
        500: {"model": ErrorDetailResponse},
    },
)
async def cancel_index_job(
    folder_id: int,
    job_id: str,
    token: str = Depends(extract_token),
    folder: Folder = Depends(get_folder_by_id),
):
    """Cancel a running background indexing job."""
    try:
        status = await job_manager.get_status(job_id)
        if not status or status.get("folder_id") != folder_id:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        cancelled = await job_manager.cancel_job(job_id)
        if not cancelled:
            return {
                "data": {"cancelled": False, "job_id": job_id},
                "message": "Job already finished — nothing to cancel",
            }
        return {
            "data": {"cancelled": True, "job_id": job_id},
            "message": "Job cancelled",
        }
    except HTTPException:
        raise
    except Exception as e:
        log_err(logger, "API_CANCEL_JOB_FAIL", e, folder_id=folder_id)
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


# Response = {data: FileIndexStatusResponse, message} — status ∈ indexed / indexing / queued / failed / not_indexed
@router.get(
    "/file/{file_id}/index/status",
    tags=[_TAG_STATUS],
    response_model=FileIndexStatusEnvelope,
    responses={
        404: {"model": ErrorDetailResponse, "description": "file does not exist or no permission"},
        500: {"model": ErrorDetailResponse},
    },
)
async def get_file_index_status(
    file_id: UUID,
    token: str = Depends(extract_token),
    file: File = Depends(get_file_by_id),
):
    """Query a single file's index status.

    Args:
        file_id: File UUID.
        token: User token (permission check).
        file: Injected by the get_file_by_id dependency.

    Returns:
        envelope: ``{data: FileIndexStatusResponse, message}``;
        status ∈ ``indexed`` / ``indexing`` / ``queued`` / ``failed`` / ``not_indexed``.
    """
    try:
        from db.fileindexdb import FileIndexDB

        record = FileIndexDB.get_by_file(file_id)
        if record and getattr(record, "status", None) == "indexed":
            payload = FileIndexStatusResponse(
                file_id=str(file_id),
                file_name=file.file_name,
                status="indexed",
                num_chunks=getattr(record, "num_chunks", None),
                indexed_at=record.indexed_at.isoformat() if getattr(record, "indexed_at", None) else None,
                job_id=None,
                progress=None,
                error=None,
            )
            return {"data": payload.model_dump(), "message": "File is indexed"}
        if record and getattr(record, "status", None) == "failed":
            payload = FileIndexStatusResponse(
                file_id=str(file_id),
                file_name=file.file_name,
                status="failed",
                num_chunks=None,
                indexed_at=None,
                job_id=None,
                progress=None,
                error=getattr(record, "error_message", None),
            )
            return {"data": payload.model_dump(), "message": "Indexing previously failed"}

        # Not in DB (or not in a terminal state) — check live running jobs
        file_id_str = str(file_id)
        for job in job_manager.list_jobs(status_filter=JobStatus.RUNNING):
            if job.get("current_file_id") == file_id_str:
                payload = FileIndexStatusResponse(
                    file_id=file_id_str,
                    file_name=file.file_name,
                    status="indexing",
                    num_chunks=None,
                    indexed_at=None,
                    job_id=job["job_id"],
                    progress={
                        "current_index": job.get("current_index", 0),
                        "total_files": job.get("total_files", 0),
                    },
                    error=None,
                )
                return {
                    "data": payload.model_dump(),
                    "message": "File is currently being indexed",
                }
            if file_id_str in (job.get("scope_file_ids") or []):
                payload = FileIndexStatusResponse(
                    file_id=file_id_str,
                    file_name=file.file_name,
                    status="queued",
                    num_chunks=None,
                    indexed_at=None,
                    job_id=job["job_id"],
                    progress={
                        "current_index": job.get("current_index", 0),
                        "total_files": job.get("total_files", 0),
                    },
                    error=None,
                )
                return {
                    "data": payload.model_dump(),
                    "message": "File is queued for indexing",
                }

        payload = FileIndexStatusResponse(
            file_id=file_id_str,
            file_name=file.file_name,
            status="not_indexed",
            num_chunks=None,
            indexed_at=None,
            job_id=None,
            progress=None,
            error=None,
        )
        return {"data": payload.model_dump(), "message": "File has not been indexed"}

    except HTTPException:
        raise
    except Exception as e:
        log_err(logger, "API_FILE_STATUS_FAIL", e, file_id=file_id)
        raise HTTPException(status_code=500, detail=f"Failed to get file status: {str(e)}")


# Deletion

# Resource = all of the folder's FileIndex + vector store table; Response = DeleteFolderIndexResponse
@router.delete(
    "/files/{folder_id}/index",
    tags=[_TAG_CLEANUP],
    responses={
        404: {"model": ErrorDetailResponse, "description": "folder does not exist"},
        500: {"model": ErrorDetailResponse},
    },
)
async def delete_folder_index_endpoint(
    folder_id: int,
    token: str = Depends(extract_token),
    folder: Folder = Depends(get_folder_by_id),
):
    """Delete an entire folder's index (keeping the folder and the files themselves)."""
    try:
        logger.info(f"API: Deleting folder index for folder_id={folder_id}")
        result = await get_rag_adapter().delete_folder_index(folder_id=folder_id, token=token)

        if result.total_files == 0:
            return {"data": result, "message": "No indexed files found in folder"}

        return {"data": result, "message": result.message}

    except HTTPException:
        raise
    except Exception as e:
        log_err(logger, "API_DELETE_FOLDER_IDX_FAIL", e, folder_id=folder_id, token=token)
        raise HTTPException(status_code=500, detail=f"Failed to delete folder index: {str(e)}")


# Response = {data: {file_id, file_name, deleted: bool}, message}
@router.delete(
    "/file/{file_id}/index",
    tags=[_TAG_CLEANUP],
    responses={
        404: {"model": ErrorDetailResponse, "description": "file does not exist"},
        500: {"model": ErrorDetailResponse},
    },
)
async def delete_single_file_index_endpoint(
    file_id: UUID,
    token: str = Depends(extract_token),
    file: File = Depends(get_file_by_id),
):
    """Delete a single file's index."""
    try:
        logger.info(f"API: Deleting index for single file_id={file_id}")
        success = await get_rag_adapter().delete_document_index(file_id=file_id, token=token)

        if not success:
            return {
                "data": {"file_id": str(file_id), "deleted": False},
                "message": "No index found for this file",
            }

        return {
            "data": {
                "file_id": str(file_id),
                "file_name": file.file_name,
                "deleted": True,
            },
            "message": f"Successfully deleted index for file: {file.file_name}",
        }

    except HTTPException:
        raise
    except Exception as e:
        log_err(logger, "API_DELETE_FILE_IDX_FAIL", e, file_id=file_id, token=token)
        raise HTTPException(status_code=500, detail=f"Failed to delete file index: {str(e)}")
