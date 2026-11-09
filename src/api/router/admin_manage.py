
"""Control Center management API — CRUD and operations for folders / files / index.

Every endpoint needs a console session (or a bearer PAT) and only reaches folders
owned by that session's token (its jti) — the same folders a REST / MCP caller with
the token can reach. Operations call the existing REST endpoint functions with that
owner key, so all security and cleanup logic stays single-source. A folder you do
not own is reported as not found (404), exactly like on the REST API.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.api.router.admin_scope import owned_folder
from src.api.router.response import CreateFolderRequest, UpdateFolderRequest
from src.auth.console_session import ConsoleIdentity, require_console_session
from src.domain.rag.query_limits import MAX_QUERY_CHARS, MIN_QUERY_CHARS
from src.log import get_api_logger

logger = get_api_logger()

router = APIRouter(tags=["Admin: Manage"], dependencies=[Depends(require_console_session)])

# Owner key for folders created in open (auth disabled) mode, where there is no token.
OPEN_MODE_OWNER = "console"

_Session = Depends(require_console_session)


class PlaygroundQuery(BaseModel):
    """Body of the console's search playground (/query and /query/stream).

    Typed so a wrong type or an out-of-range value is a 422, not a 500 deep in the
    engine (``query: 123`` used to crash on ``.strip()``, ``top_k: 500`` reached the
    retriever). Ranges match the REST QueryRequest.
    """
    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., max_length=MAX_QUERY_CHARS)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    similarity_cutoff: Optional[float] = Field(None, ge=0.0, le=1.0)
    auto_merge: Optional[bool] = None
    trace: bool = False
    answer: bool = False

    @field_validator("query")
    @classmethod
    def _strip_min(cls, v: str) -> str:
        v = v.strip()
        if len(v) < MIN_QUERY_CHARS:
            raise ValueError(f"query too short (min {MIN_QUERY_CHARS} chars)")
        return v


def _query_failed(e: Exception, folder_id: int) -> HTTPException:
    """Map a retrieval failure to an HTTP error (context-window overflow → 422)."""
    from src.domain.rag.query_limits import QUERY_TOO_LONG_DETAIL, is_input_too_long_error
    if is_input_too_long_error(e):
        return HTTPException(422, QUERY_TOO_LONG_DETAIL)
    logger.warning(f"[ADMIN] query failed for folder {folder_id}: {e}")
    return HTTPException(500, f"Query failed: {e}")


def _owner(folder_id: int, ident: ConsoleIdentity):
    """Load a folder the session owns and return (folder, owner_key) — 404 otherwise."""
    folder = owned_folder(folder_id, ident)
    return folder, folder.user_token


@router.post("/api/admin/manage/folders")
async def admin_create_folder(
    body: CreateFolderRequest = Body(..., examples=[{"name": "docs", "description": ""}]),
    ident: ConsoleIdentity = _Session,
):
    """Create a folder owned by the signed-in token (reachable via REST / MCP with that token).

    Any owner field in the body is ignored: the owner is always the session.
    """
    from src.api.router.folder_api import create_folder
    return await create_folder(
        CreateFolderRequest(name=body.name, description=body.description),
        user_token=ident.owner_key or OPEN_MODE_OWNER,
    )


@router.patch("/api/admin/manage/folders/{folder_id}")
async def admin_update_folder(folder_id: int, body: UpdateFolderRequest = Body(...),
                             ident: ConsoleIdentity = _Session):
    """Rename / change description (only patches the fields provided)."""
    _, token = _owner(folder_id, ident)
    from src.api.router.folder_api import update_folder_api
    return await update_folder_api(
        folder_id,
        UpdateFolderRequest(name=body.name, description=body.description),
        user_token=token,
    )


@router.delete("/api/admin/manage/folders/{folder_id}")
async def admin_delete_folder(folder_id: int, ident: ConsoleIdentity = _Session):
    """Delete a folder (reuses the existing endpoint: cancel in-flight jobs first, then soft-delete files and vectors)."""
    _, token = _owner(folder_id, ident)
    from src.api.router.folder_api import delete_folder
    return await delete_folder(folder_id, user_token=token)


@router.post("/api/admin/manage/folders/{folder_id}/files")
async def admin_upload_file(
    folder_id: int,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    auto_index: bool = Form(True),
    ident: ConsoleIdentity = _Session,
):
    """Upload a file to the folder (auto_index=True runs the existing background indexing flow)."""
    _, token = _owner(folder_id, ident)
    from src.api.router.file_api import upload_file
    return await upload_file(
        folder_id, file=file, description=description, tags=None,
        auto_index=auto_index, user_token=token,
    )


@router.get("/api/admin/manage/folders/{folder_id}/files/{file_id}/download")
async def admin_download_file(folder_id: int, file_id: UUID, ident: ConsoleIdentity = _Session):
    _, token = _owner(folder_id, ident)
    from src.api.router.file_api import download_file
    return await download_file(folder_id, file_id, user_token=token)


@router.delete("/api/admin/manage/folders/{folder_id}/files/{file_id}")
async def admin_delete_file(folder_id: int, file_id: UUID, ident: ConsoleIdentity = _Session):
    """Delete a file (the existing endpoint also clears the file's vectors and index records)."""
    _, token = _owner(folder_id, ident)
    from src.api.router.file_api import delete_file
    return await delete_file(folder_id, file_id, user_token=token)


@router.post("/api/admin/manage/folders/{folder_id}/index")
async def admin_index_folder(
    folder_id: int,
    body: dict = Body(default={}, examples=[{"skip_existing": True}]),
    ident: ConsoleIdentity = _Session,
):
    """Start a background indexing job for the whole folder.

    skip_existing=True is incremental (only files without an index record).
    skip_existing=False routes to the reindex endpoint (delete old records, then
    rerun); a plain index cannot rebuild because the per-file content_hash
    short-circuit skips unchanged files, so settings changes would have no effect.
    """
    folder, token = _owner(folder_id, ident)
    from src.api.router.response import IndexRequest
    skip_existing = bool(body.get("skip_existing", True))
    if skip_existing:
        from src.api.router.rag_indexing import index_folder_endpoint
        return await index_folder_endpoint(
            folder_id, IndexRequest(), skip_existing=True, token=token, folder=folder)
    from src.api.router.rag_indexing import reindex_folder_endpoint
    return await reindex_folder_endpoint(
        folder_id, IndexRequest(), skip_existing=False, token=token, folder=folder)


def _fts_mode() -> str:
    from src.domain.rag.hybrid_search import configured_text_search_config
    return configured_text_search_config()


@router.get("/api/admin/manage/folders/{folder_id}/fts")
async def admin_fts_status(folder_id: int, ident: ConsoleIdentity = _Session):
    """Which segmenter built the folder's keyword index, and whether it needs a rebuild.

    ``needs_rebuild`` turns true after ``rag.retrieval.segmenter`` changes: the
    query side already uses the new segmenter, the stored tsvector does not.
    """
    folder, _ = _owner(folder_id, ident)
    from src.domain.rag.text_segmenter import configured_segmenter_name
    configured = configured_segmenter_name()
    if _fts_mode() != "simple":
        return {"data": {"folder_id": folder_id, "text_search_config": _fts_mode(),
                         "configured": None, "indexed_with": None, "needs_rebuild": False}}
    from starlette.concurrency import run_in_threadpool
    from db.db import get_engine
    from src.domain.rag.vector_store_manager import VectorStoreManager, fts_status
    table_name = VectorStoreManager.physical_table_name(folder_id, folder.vector_table_uuid)
    try:
        status = await run_in_threadpool(fts_status, get_engine(), table_name, configured)
    except Exception as e:  # noqa: BLE001 — status is advisory; never break the page
        logger.warning(f"[ADMIN] fts status failed for folder {folder_id}: {e}")
        status = {"indexed_with": None, "configured": configured, "needs_rebuild": False}
    return {"data": {"folder_id": folder_id, "text_search_config": "simple", **status}}


@router.post("/api/admin/manage/folders/{folder_id}/rebuild-fts", status_code=202)
async def admin_rebuild_fts(folder_id: int, ident: ConsoleIdentity = _Session):
    """Start a background job that re-segments the folder's full-text search vector.

    Uses the configured segmenter (``rag.retrieval.segmenter``: ckip / jieba /
    bigram) and, once every file is done, stamps the table with it. Needed after
    switching segmenters, and for folders indexed before Python-side segmentation
    existed (llama-index's native ``to_tsvector('simple', text)`` cannot segment
    space-less Chinese). The job re-segments every stored chunk file by file and
    overwrites the tsvector in place — no re-embedding, no re-parsing. 409 unless
    FTS uses the ``simple`` config (the segmented mode), or while another job holds
    the folder.

    Returns the job record at once (202); progress (files done / total), cancel and
    the final summary go through the usual job endpoints, like an indexing job. It
    used to run inside the request, and a 5.6k-chunk folder outlived the HTTP timeout.
    """
    folder, _ = _owner(folder_id, ident)

    # Only meaningful in segmented mode; other FTS configs keep the native tsvector.
    if _fts_mode() != "simple":
        raise HTTPException(
            409, "FTS is not in segmented mode "
            "(rag.retrieval.text_search_config != 'simple'); nothing to rebuild")

    from db.db import get_engine
    from src.adapter.fts_rebuild import rebuild_fts_by_file
    from src.domain.exceptions import ConflictError
    from src.domain.rag.index_job_manager import IndexingJobManager, JOB_KIND_REBUILD_FTS
    from src.domain.rag.vector_store_manager import VectorStoreManager

    table_name = VectorStoreManager.physical_table_name(folder_id, folder.vector_table_uuid)

    def _work(progress, should_stop):
        from src.domain.rag.text_segmenter import get_segmenter
        return rebuild_fts_by_file(get_engine(), table_name, get_segmenter(),
                                   progress, should_stop)

    try:
        job = await IndexingJobManager.get_instance().start_maintenance_job(
            folder_id=folder_id, kind=JOB_KIND_REBUILD_FTS, work=_work,
            message="Rebuilding full-text search")
    except ConflictError as e:
        raise HTTPException(409, e.message)
    return {"data": job, "message": "Full-text search rebuild started"}


@router.get("/api/admin/manage/folders/{folder_id}/jobs/{job_id}")
async def admin_job_status(folder_id: int, job_id: str, ident: ConsoleIdentity = _Session):
    """Live state of one job on the folder (indexing or maintenance) — for progress polling."""
    _owner(folder_id, ident)
    from src.domain.rag.index_job_manager import IndexingJobManager
    st = await IndexingJobManager.get_instance().get_status(job_id)
    if not st or st.get("folder_id") != folder_id:
        from db.indexjobdb import IndexJobDB
        row = IndexJobDB.get_by_id(job_id)
        if not row or row.get("folder_id") != folder_id:
            raise HTTPException(404, f"Job {job_id} not found")
        st = {k: (str(v) if k == "job_id" else v) for k, v in row.items()}
    return {"data": st, "message": "ok"}


def _files_in_folder(folder_id: int, file_ids) -> list:
    """Resolve file ids that must all live in ``folder_id`` — 404 on the first that doesn't.

    The folder is owned by the session; without this check a folder you own could be
    used to reindex or cancel a file in someone else's folder.
    """
    from db.filedb import FileDB
    out = []
    for fid in file_ids:
        rows = FileDB.get(id=fid)
        if not rows or rows[0].folder_id != folder_id:
            raise HTTPException(404, f"File {fid} not found")
        out.append(str(rows[0].id))
    return out


async def _start_retry(folder_id: int, token: str, file_ids: list) -> dict:
    """Queue a forced re-index of ``file_ids`` as ONE background job under the folder lock."""
    from src.adapter.rag import get_rag_adapter
    from src.domain.rag.index_job_manager import IndexingJobManager
    return await IndexingJobManager.get_instance().start_indexing_files(
        get_rag_adapter(), file_ids=file_ids, folder_id=folder_id, token=token,
        chunk_size=None, chunk_overlap=None, force=True)


@router.post("/api/admin/manage/folders/{folder_id}/files/{file_id}/retry", status_code=202)
async def admin_retry_file(folder_id: int, file_id: UUID, ident: ConsoleIdentity = _Session):
    """Retry (re-index) a single file as a background job.

    force bypasses the content_hash short-circuit, so this also rebuilds an indexed file.
    It used to index inline in the request and outside the folder lock, so a large file
    timed out the HTTP call and "Reindex selected" raced several writers on one table.
    Now it queues behind any running job on the folder and returns the job (202).
    """
    _, token = _owner(folder_id, ident)
    ids = _files_in_folder(folder_id, [file_id])
    job = await _start_retry(folder_id, token, ids)
    return {"data": job, "message": "Reindex queued" if job.get("queued") else "Reindex started"}


@router.post("/api/admin/manage/folders/{folder_id}/retry", status_code=202)
async def admin_retry_files(folder_id: int, body: dict = Body(..., examples=[{"file_ids": ["…"]}]),
                            ident: ConsoleIdentity = _Session):
    """Retry several files of the folder as one background job (the console's "Reindex selected")."""
    _, token = _owner(folder_id, ident)
    raw = body.get("file_ids")
    if not isinstance(raw, list) or not raw or len(raw) > 1000:
        raise HTTPException(422, "file_ids must be a non-empty list (at most 1000)")
    try:
        parsed = list(dict.fromkeys(UUID(str(x)) for x in raw))
    except (ValueError, TypeError):
        raise HTTPException(422, "file_ids must be UUIDs")
    ids = _files_in_folder(folder_id, parsed)
    job = await _start_retry(folder_id, token, ids)
    return {"data": job, "message": f"Reindex of {len(ids)} file(s) "
                                    + ("queued" if job.get("queued") else "started")}


@router.post("/api/admin/manage/folders/{folder_id}/files/{file_id}/cancel")
async def admin_cancel_file(folder_id: int, file_id: UUID, ident: ConsoleIdentity = _Session):
    """Cancel indexing of one file without stopping the rest of its job.

    A queued file is dropped from its batch; the only file of a job cancels that job; a
    file inside a running multi-file job is stopped at its next checkpoint and recorded
    as "cancelled". 409 when no live job has the file.
    """
    _owner(folder_id, ident)
    fid = _files_in_folder(folder_id, [file_id])[0]
    from src.domain.rag.index_job_manager import IndexingJobManager
    outcome = await IndexingJobManager.get_instance().cancel_file(fid, folder_id)
    if outcome is None:
        raise HTTPException(409, "This file is not queued or being indexed")
    msg = {"dequeued": "Removed from the queue",
           "cancelled_job": "Indexing cancelled",
           "flagged": "Cancelling — the file stops at its next checkpoint"}[outcome]
    return {"data": {"file_id": fid, "outcome": outcome}, "message": msg}


@router.post("/api/admin/manage/folders/{folder_id}/query")
async def admin_query_folder(folder_id: int,
                             body: PlaygroundQuery = Body(..., examples=[{"query": "…"}]),
                             ident: ConsoleIdentity = _Session):
    """Run one retrieval against the folder (query test bench). Returns matched chunks + scores.

    trace=true: run the retrieval trace (breaking out vector / BM25 / hybrid /
    rerank ranks) and build results from the trace's final hits (without rerunning
    a normal query).
    """
    folder, token = _owner(folder_id, ident)
    q = body.query

    if body.trace:
        from src.adapter.rag import get_rag_adapter
        from src.api.router.rag_query import _resolve_retrieval_defaults
        d = _resolve_retrieval_defaults()
        try:
            trace = await get_rag_adapter().query_trace(
                query=q, folder_id=folder_id, token=token,
                top_k=body.top_k or d["top_k"],
                similarity_cutoff=(body.similarity_cutoff if body.similarity_cutoff is not None
                                   else d["similarity_cutoff"]),
                sparse_top_k=d["sparse_top_k"], hybrid_alpha=d["hybrid_alpha"])
        except HTTPException:
            raise
        except Exception as e:
            raise _query_failed(e, folder_id)
        results = [{"text": r["text"], "score": r["final_score"], "metadata": r["metadata"]}
                   for r in trace["results"]]
        resp = {"data": {"query": q, "results": results,
                         "total_results": len(results), "trace": trace}, "message": "ok"}
    else:
        from src.api.router.rag_query import query_rag
        from src.api.router.response import QueryRequest
        from fastapi import Request as _Req
        req = QueryRequest(
            query=q, folder_name=folder.name,
            top_k=body.top_k, similarity_cutoff=body.similarity_cutoff,
            auto_merge=body.auto_merge,
        )
        scope = {"type": "http", "headers": [], "method": "POST", "path": "/", "query_string": b""}
        resp = await query_rag(_Req(scope), req, token=token)
    # answer=true: follow retrieval with CRAG answer generation.
    if body.answer:
        try:
            # Two blocking LLM calls: keep them off the event loop.
            from starlette.concurrency import run_in_threadpool
            gen = await run_in_threadpool(_generate_answer, q, resp["data"].get("results", []))
            resp["data"]["answer"] = gen.get("answer")
            resp["data"]["rag_gate"] = {k: gen[k] for k in
                                        ("confidence", "kept", "dropped", "note", "citations",
                                         "invalid_citations", "grading_error") if k in gen}
        except Exception as e:
            logger.warning(f"[ADMIN] answer generation failed: {e}")
            resp["data"]["answer"] = None
    return resp


def _answer_llm():
    """(client, model) for the answer LLM (rag.answer_llm over rag.llm)."""
    from src.domain.rag.context_generator import ContextGenerator
    from src.domain.rag.llm_client import model_name, role_llm
    llm_cfg = role_llm("answer_llm")  # rag.answer_llm over rag.llm
    client = ContextGenerator._create_sync_client(llm_cfg)
    return client, model_name(llm_cfg)


def _close_quietly(client) -> None:
    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass


def _generate_answer(query: str, results: list) -> dict:
    """CRAG (Corrective RAG) answer generation — see src/domain/rag/answer.py.

    Returns {answer, confidence (high/medium/low/none/unverified), kept, dropped,
    note, citations, invalid_citations}.
    """
    from src.domain.rag import answer as crag
    if not results:
        return crag.answer(None, "", query, results)
    client, model = _answer_llm()
    try:
        return crag.answer(client, model, query, results)
    finally:
        _close_quietly(client)


def _crag_grade(query: str, results: list) -> dict:
    """CRAG relevance grading for the streaming path (one LLM call).

    Returns {top, contexts, kept, dropped, grading}: ``contexts`` are the kept
    chunks with their original result numbers. A grading failure keeps every
    chunk but is reported (the answer is then labelled unverified).
    """
    from src.domain.rag import answer as crag
    top = list(results[:crag.MAX_CONTEXTS])
    if not top:
        return {"top": [], "contexts": [], "kept": [], "dropped": 0,
                "grading": crag.Grading([])}
    client, model = _answer_llm()
    try:
        grading = crag.grade(client, model, query, crag.grade_texts(top))
    finally:
        _close_quietly(client)
    if not grading.ok:
        logger.warning(f"[CRAG] relevance grading failed ({grading.reason}); answer unverified")
    contexts = crag.contexts_for(top, grading.grades)
    kept = [r for r, g in zip(top, grading.grades) if g]
    return {"top": top, "contexts": contexts, "kept": kept,
            "dropped": len(top) - len(contexts), "grading": grading}


def _generate_stream(query: str, contexts: list):
    """Blocking generator yielding answer text pieces from the LLM (stream=True).

    Runs in a worker thread; the async endpoint bridges it to the event loop.
    Full-width citations are normalised per piece where possible; the final
    cleaned answer is sent on the ``done`` frame.
    """
    from src.domain.rag import answer as crag
    client, model = _answer_llm()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": crag.answer_prompt(query, contexts)}],
            max_tokens=2048, temperature=0.2, stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                yield crag.normalise_citations(delta)
    finally:
        _close_quietly(client)


async def _stream_answer_events(query: str, results: list):
    """Async generator of SSE ``data:`` frames for a streamed CRAG answer.

    Frames: {type: meta|token|done|error}. Retrieval already happened; this grades
    the chunks, gates on relevance, then streams the generation token by token.
    The ``done`` frame carries the cleaned final answer (citations normalised and
    validated, bold markers removed) and replaces what the tokens built up.
    """
    import asyncio
    import json
    from starlette.concurrency import run_in_threadpool
    from src.domain.rag import answer as crag

    def _sse(obj) -> str:
        # default=str: result metadata carries non-JSON types (UUID node_id).
        return f"data: {json.dumps(obj, ensure_ascii=False, default=str)}\n\n"

    if not results:
        yield _sse({"type": "meta", "confidence": crag.CONF_NONE, "kept": 0, "dropped": 0,
                    "note": "No chunks retrieved"})
        yield _sse({"type": "done", "answer": None})
        return

    graded = await run_in_threadpool(_crag_grade, query, results)
    contexts, dropped, grading = graded["contexts"], graded["dropped"], graded["grading"]
    meta = {"type": "meta", "kept": len(contexts), "dropped": dropped,
            "confidence": crag.confidence(grading, len(contexts))}
    if not grading.ok:
        meta["grading_error"] = grading.reason
    yield _sse(meta)

    if not contexts:
        yield _sse({"type": "token", "text": crag.NO_BASIS_MESSAGE})
        yield _sse({"type": "done", "answer": crag.NO_BASIS_MESSAGE, "confidence": crag.CONF_LOW,
                    "note": "CRAG gate: no retrieved chunk graded relevant"})
        return

    # Bridge the blocking token generator to the event loop via a queue.
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    SENTINEL = object()

    def _produce():
        try:
            for piece in _generate_stream(query, contexts):
                loop.call_soon_threadsafe(queue.put_nowait, piece)
        except Exception as e:  # noqa: BLE001 - surface generation errors to the client
            loop.call_soon_threadsafe(queue.put_nowait, {"__err__": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    task = asyncio.create_task(run_in_threadpool(_produce))
    full: list[str] = []
    failed = False
    try:
        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            if isinstance(item, dict) and "__err__" in item:
                yield _sse({"type": "error", "error": item["__err__"]})
                failed = True
                break
            full.append(item)
            yield _sse({"type": "token", "text": item})
    finally:
        await task
    if failed:
        return
    yield _sse({"type": "done", **crag.finish_stream("".join(full), contexts, grading, query)})


@router.post("/api/admin/manage/folders/{folder_id}/query/stream")
async def admin_query_folder_stream(folder_id: int,
                                    body: PlaygroundQuery = Body(..., examples=[{"query": "…"}]),
                                    ident: ConsoleIdentity = _Session):
    """Retrieve, then stream a CRAG answer over Server-Sent Events (text/event-stream).

    Retrieval and relevance grading complete first (a `sources` frame carries the
    matched chunks); only the final generation streams token by token. Same
    ownership rules as the non-streaming /query endpoint.
    """
    import json
    from starlette.responses import StreamingResponse
    folder, token = _owner(folder_id, ident)
    q = body.query

    from src.api.router.rag_query import query_rag
    from src.api.router.response import QueryRequest
    from fastapi import Request as _Req
    req = QueryRequest(
        query=q, folder_name=folder.name,
        top_k=body.top_k, similarity_cutoff=body.similarity_cutoff,
        auto_merge=body.auto_merge)
    scope = {"type": "http", "headers": [], "method": "POST", "path": "/", "query_string": b""}
    resp = await query_rag(_Req(scope), req, token=token)
    results = resp["data"].get("results", [])

    async def _gen():
        # Headers are already sent once streaming starts, so an exception here can't
        # become an HTTP error — surface it as an `error` frame instead of a silent
        # close. default=str: result metadata carries non-JSON types (UUID node_id).
        try:
            yield (f"data: {json.dumps({'type': 'sources', 'results': results, 'total': len(results)}, ensure_ascii=False, default=str)}\n\n")
            async for frame in _stream_answer_events(q, results):
                yield frame
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ADMIN] query/stream failed for folder {folder_id}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

    # X-Accel-Buffering: no keeps a reverse proxy (nginx) from buffering the stream.
    return StreamingResponse(
        _gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/api/admin/manage/folders/{folder_id}/jobs/{job_id}/cancel")
async def admin_cancel_job(folder_id: int, job_id: str, ident: ConsoleIdentity = _Session):
    """Cancel a running indexing job."""
    folder, token = _owner(folder_id, ident)
    from src.api.router.rag_indexing import cancel_index_job
    return await cancel_index_job(folder_id, job_id, token=token, folder=folder)
