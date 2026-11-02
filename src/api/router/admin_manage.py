
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

from src.api.router.admin_scope import owned_folder
from src.auth.console_session import ConsoleIdentity, require_console_session
from src.log import get_api_logger

logger = get_api_logger()

router = APIRouter(tags=["Admin: Manage"], dependencies=[Depends(require_console_session)])

# Owner key for folders created in open (auth disabled) mode, where there is no token.
OPEN_MODE_OWNER = "console"

_Session = Depends(require_console_session)


def _owner(folder_id: int, ident: ConsoleIdentity):
    """Load a folder the session owns and return (folder, owner_key) — 404 otherwise."""
    folder = owned_folder(folder_id, ident)
    return folder, folder.user_token


@router.post("/api/admin/manage/folders")
async def admin_create_folder(
    body: dict = Body(..., examples=[{"name": "docs", "description": ""}]),
    ident: ConsoleIdentity = _Session,
):
    """Create a folder owned by the signed-in token (reachable via REST / MCP with that token)."""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(422, "name is required")
    from src.api.router.folder_api import create_folder
    from src.api.router.response import CreateFolderRequest
    return await create_folder(
        CreateFolderRequest(name=name, description=body.get("description")),
        user_token=ident.owner_key or OPEN_MODE_OWNER,
    )


@router.patch("/api/admin/manage/folders/{folder_id}")
async def admin_update_folder(folder_id: int, body: dict = Body(...),
                             ident: ConsoleIdentity = _Session):
    """Rename / change description (only patches the fields provided)."""
    _, token = _owner(folder_id, ident)
    from src.api.router.folder_api import update_folder_api
    from src.api.router.response import UpdateFolderRequest
    return await update_folder_api(
        folder_id,
        UpdateFolderRequest(name=body.get("name"), description=body.get("description")),
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
    from src.config.config_manager import Config
    ret = getattr(getattr(Config.get_config_model(), "rag", None), "retrieval", None)
    return getattr(ret, "text_search_config", "simple") or "simple"


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


@router.post("/api/admin/manage/folders/{folder_id}/rebuild-fts")
async def admin_rebuild_fts(folder_id: int, ident: ConsoleIdentity = _Session):
    """Re-segment the folder's full-text search vector (`text_search_tsv`).

    Uses the configured segmenter (``rag.retrieval.segmenter``: ckip / jieba /
    bigram) and stamps the table with it. Needed after switching segmenters,
    and for folders indexed before Python-side segmentation existed. No
    re-embedding and no re-parsing — only the search vector changes. No-op (409)
    unless FTS uses the ``simple`` config (the segmented mode).

    Blocking (sync DB + segmentation), so it runs in a threadpool off the loop.
    """
    folder, _ = _owner(folder_id, ident)

    # Only meaningful in segmented mode; other FTS configs keep the native tsvector.
    if _fts_mode() != "simple":
        raise HTTPException(
            409, "FTS is not in segmented mode "
            "(rag.retrieval.text_search_config != 'simple'); nothing to rebuild")

    from starlette.concurrency import run_in_threadpool
    from db.db import get_engine
    from src.domain.rag.vector_store_manager import (
        VectorStoreManager, rebuild_text_search_tsv)
    from src.domain.rag.text_segmenter import get_segmenter

    table_name = VectorStoreManager.physical_table_name(
        folder_id, folder.vector_table_uuid)
    segmenter = get_segmenter()

    def _work() -> int:
        return rebuild_text_search_tsv(get_engine(), table_name, segmenter)

    try:
        rows = await run_in_threadpool(_work)
    except Exception as e:
        logger.warning(f"[ADMIN] rebuild-fts failed for folder {folder_id}: {e}")
        raise HTTPException(500, f"Rebuild failed: {e}")
    name = getattr(segmenter, "name", "configured")
    return {"data": {"folder_id": folder_id, "rows_updated": rows, "segmenter": name},
            "message": f"Rebuilt full-text search for {rows} chunk(s) with {name}"}


@router.post("/api/admin/manage/folders/{folder_id}/files/{file_id}/retry")
async def admin_retry_file(folder_id: int, file_id: UUID, ident: ConsoleIdentity = _Session):
    """Retry indexing a single file (force bypasses the content_hash short-circuit; one-click rerun for failed files)."""
    _, token = _owner(folder_id, ident)
    from db.filedb import FileDB
    from src.domain.rag.dto import FileRequest
    from src.adapter.rag import get_rag_adapter
    rows = FileDB.get(id=file_id)
    # The file must live in this (owned) folder: otherwise a folder you own could be
    # used to reindex a file in someone else's folder.
    if not rows or rows[0].folder_id != folder_id:
        raise HTTPException(404, f"File {file_id} not found")
    f = rows[0]
    req = FileRequest(
        id=f.id, file_name=f.file_name, file_path=f.file_path,
        folder_id=f.folder_id, mime_type=f.mime_type, file_size=f.file_size,
        content_hash=getattr(f, "content_hash", None))
    result = await get_rag_adapter().index_document(
        file_record=req, token=token, force=True)
    return {"data": result.model_dump(), "message": "Reindexed"}


@router.post("/api/admin/manage/folders/{folder_id}/query")
async def admin_query_folder(folder_id: int, body: dict = Body(..., examples=[{"query": "…"}]),
                             ident: ConsoleIdentity = _Session):
    """Run one retrieval against the folder (query test bench). Returns matched chunks + scores.

    trace=true: run the retrieval trace (breaking out vector / BM25 / hybrid /
    rerank ranks) and build results from the trace's final hits (without rerunning
    a normal query).
    """
    folder, token = _owner(folder_id, ident)
    q = (body.get("query") or "").strip()
    if len(q) < 2:
        raise HTTPException(422, "query too short (min 2 chars)")

    if body.get("trace"):
        from src.adapter.rag import get_rag_adapter
        from src.api.router.rag_query import _resolve_retrieval_defaults
        d = _resolve_retrieval_defaults()
        trace = await get_rag_adapter().query_trace(
            query=q, folder_id=folder_id, token=token,
            top_k=body.get("top_k") or d["top_k"],
            similarity_cutoff=body.get("similarity_cutoff") or d["similarity_cutoff"],
            sparse_top_k=d["sparse_top_k"], hybrid_alpha=d["hybrid_alpha"])
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
            top_k=body.get("top_k"), similarity_cutoff=body.get("similarity_cutoff"),
            auto_merge=body.get("auto_merge"),
        )
        scope = {"type": "http", "headers": [], "method": "POST", "path": "/", "query_string": b""}
        resp = await query_rag(_Req(scope), req, token=token)
    # answer=true: follow retrieval with CRAG answer generation.
    if body.get("answer"):
        try:
            gen = _generate_answer(q, resp["data"].get("results", []))
            resp["data"]["answer"] = gen.get("answer")
            resp["data"]["rag_gate"] = {k: gen[k] for k in
                                        ("confidence", "kept", "dropped", "note") if k in gen}
        except Exception as e:
            logger.warning(f"[ADMIN] answer generation failed: {e}")
            resp["data"]["answer"] = None
    return resp


def _generate_answer(query: str, results: list) -> dict:
    """CRAG (Corrective RAG) answer generation: grade each retrieved chunk for
    relevance, generate from the relevant ones only, and decline to answer when
    none are relevant (avoids hallucinating on irrelevant content).

    Returns {answer, confidence (high/medium/low/none), kept, dropped, note}.
    """
    if not results:
        return {"answer": None, "confidence": "none", "kept": 0, "dropped": 0,
                "note": "No chunks retrieved"}
    import json
    import re
    from src.config.config_manager import Config
    from src.domain.rag.context_generator import ContextGenerator
    llm_cfg = Config.get_config_model().rag.llm
    client = ContextGenerator._create_sync_client(llm_cfg)
    model = getattr(llm_cfg, "azure_deployment", None) or llm_cfg.model
    top = results[:6]

    # CRAG step 1: relevance grading (single LLM call returning a boolean array)
    grades = [True] * len(top)
    try:
        gprompt = (
            "Grade whether each numbered context can help answer the question. "
            "Reply ONLY a JSON array of booleans (one per context, true = relevant).\n\n"
            + "\n\n".join(f"[{i+1}] {r['text'][:600]}" for i, r in enumerate(top))
            + f"\n\nQuestion: {query}\n\nJSON:")
        gr = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": gprompt}],
            max_tokens=1024, temperature=0)
        m = re.search(r"\[.*?\]", (gr.choices[0].message.content or ""), re.S)
        if m:
            arr = json.loads(m.group(0))
            grades = [bool(x) for x in arr][:len(top)] + [False] * (len(top) - len(arr))
    except Exception as e:
        logger.debug(f"[CRAG] grading skipped, using all context: {e}")

    kept = [r for r, g in zip(top, grades) if g]
    dropped = len(top) - len(kept)

    # CRAG gate: all irrelevant -> don't force an answer
    if not kept:
        try:
            client.close()
        except Exception:
            pass
        return {"answer": "The currently indexed content has no sufficient basis "
                          "to answer this question. Consider adding relevant "
                          "documents or rephrasing the question.",
                "confidence": "low", "kept": 0, "dropped": dropped,
                "note": "CRAG gate: no retrieved chunk graded relevant"}

    # CRAG step 2: generate from relevant chunks only (citations marked as [n])
    ctx = "\n\n".join(f"[{i+1}] {r['text'][:1200]}" for i, r in enumerate(kept))
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below. Cite sources inline as [n]. If the context lacks the answer, say so.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:")
    # Reasoning models (e.g. gpt-oss): reasoning tokens consume max_tokens; too
    # small a budget returns content None (finish_reason=length) and the answer
    # silently becomes empty. Allow 2048.
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=2048, temperature=0.2)
    ans = r.choices[0].message.content
    finish = getattr(r.choices[0], "finish_reason", None)
    if not (ans or "").strip():
        # Still empty (usually reasoning exhausted the budget) -> return a clear message rather than silently empty.
        logger.warning(f"[CRAG] empty generation (finish={finish}) for query={query!r}")
        try:
            client.close()
        except Exception:
            pass
        return {"answer": "Relevant content was found, but answer generation did "
                          "not complete (model output was truncated). Please try "
                          "again or use a more specific question.",
                "confidence": "low", "kept": len(kept), "dropped": dropped,
                "note": f"generation returned empty (finish={finish})"}
    # gpt-oss emits OpenAI-style citations 【n†...】 that the UI can't render;
    # normalize to [n] to match the result cards.
    ans = re.sub(r"【\s*(\d+)\s*†[^】]*】", r"[\1]", ans)
    try:
        client.close()
    except Exception:
        pass
    return {"answer": ans,
            "confidence": "high" if len(kept) >= 2 else "medium",
            "kept": len(kept), "dropped": dropped,
            "note": f"Answered from {len(kept)} relevant chunk(s)"}


def _crag_grade(query: str, results: list) -> dict:
    """CRAG relevance grading (one LLM call): return {top, kept, dropped}.

    Shared by the streaming answer path. `kept` are the chunks graded relevant to
    the question; on any grading error every chunk is kept (fail-open, same as the
    non-streaming path).
    """
    import json
    import re
    from src.config.config_manager import Config
    from src.domain.rag.context_generator import ContextGenerator
    top = results[:6]
    if not top:
        return {"top": [], "kept": [], "dropped": 0}
    llm_cfg = Config.get_config_model().rag.llm
    client = ContextGenerator._create_sync_client(llm_cfg)
    model = getattr(llm_cfg, "azure_deployment", None) or llm_cfg.model
    grades = [True] * len(top)
    try:
        gprompt = (
            "Grade whether each numbered context can help answer the question. "
            "Reply ONLY a JSON array of booleans (one per context, true = relevant).\n\n"
            + "\n\n".join(f"[{i+1}] {r['text'][:600]}" for i, r in enumerate(top))
            + f"\n\nQuestion: {query}\n\nJSON:")
        gr = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": gprompt}],
            max_tokens=1024, temperature=0)
        m = re.search(r"\[.*?\]", (gr.choices[0].message.content or ""), re.S)
        if m:
            arr = json.loads(m.group(0))
            grades = [bool(x) for x in arr][:len(top)] + [False] * (len(top) - len(arr))
    except Exception as e:
        logger.debug(f"[CRAG] grading skipped, using all context: {e}")
    finally:
        try:
            client.close()
        except Exception:
            pass
    kept = [r for r, g in zip(top, grades) if g]
    return {"top": top, "kept": kept, "dropped": len(top) - len(kept)}


def _generate_stream(query: str, kept: list):
    """Blocking generator yielding answer text pieces from the LLM (stream=True).

    Runs in a worker thread; the async endpoint bridges it to the event loop.
    Citations are normalized [n] on the fly (gpt-oss emits 【n†…】).
    """
    import re
    from src.config.config_manager import Config
    from src.domain.rag.context_generator import ContextGenerator
    llm_cfg = Config.get_config_model().rag.llm
    client = ContextGenerator._create_sync_client(llm_cfg)
    model = getattr(llm_cfg, "azure_deployment", None) or llm_cfg.model
    ctx = "\n\n".join(f"[{i+1}] {r['text'][:1200]}" for i, r in enumerate(kept))
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below. Cite sources inline as [n]. If the context lacks the answer, say so.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:")
    try:
        stream = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=2048, temperature=0.2, stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0].delta, "content", None)
            if delta:
                # Normalize gpt-oss citations to [n]; safe on partial fragments.
                yield re.sub(r"【\s*(\d+)\s*†[^】]*】", r"[\1]", delta)
    finally:
        try:
            client.close()
        except Exception:
            pass


async def _stream_answer_events(query: str, results: list):
    """Async generator of SSE ``data:`` frames for a streamed CRAG answer.

    Frames: {type: meta|token|done|error}. Retrieval already happened; this grades
    the chunks, gates on relevance, then streams the generation token by token.
    """
    import asyncio
    import json
    from starlette.concurrency import run_in_threadpool

    def _sse(obj) -> str:
        # default=str: result metadata carries non-JSON types (UUID node_id).
        return f"data: {json.dumps(obj, ensure_ascii=False, default=str)}\n\n"

    if not results:
        yield _sse({"type": "meta", "confidence": "none", "kept": 0, "dropped": 0,
                    "note": "No chunks retrieved"})
        yield _sse({"type": "done", "answer": None})
        return

    graded = await run_in_threadpool(_crag_grade, query, results)
    kept, dropped = graded["kept"], graded["dropped"]
    yield _sse({"type": "meta", "kept": len(kept), "dropped": dropped,
                "confidence": "high" if len(kept) >= 2 else ("medium" if kept else "low")})

    if not kept:
        msg = ("The currently indexed content has no sufficient basis to answer this "
               "question. Consider adding relevant documents or rephrasing the question.")
        yield _sse({"type": "token", "text": msg})
        yield _sse({"type": "done", "answer": msg,
                    "note": "CRAG gate: no retrieved chunk graded relevant"})
        return

    # Bridge the blocking token generator to the event loop via a queue.
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    SENTINEL = object()

    def _produce():
        try:
            for piece in _generate_stream(query, kept):
                loop.call_soon_threadsafe(queue.put_nowait, piece)
        except Exception as e:  # noqa: BLE001 - surface generation errors to the client
            loop.call_soon_threadsafe(queue.put_nowait, {"__err__": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    task = asyncio.create_task(run_in_threadpool(_produce))
    full: list[str] = []
    try:
        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            if isinstance(item, dict) and "__err__" in item:
                yield _sse({"type": "error", "error": item["__err__"]})
                break
            full.append(item)
            yield _sse({"type": "token", "text": item})
    finally:
        await task
    answer = "".join(full)
    yield _sse({"type": "done", "answer": answer or None,
                "note": f"Answered from {len(kept)} relevant chunk(s)"})


@router.post("/api/admin/manage/folders/{folder_id}/query/stream")
async def admin_query_folder_stream(folder_id: int, body: dict = Body(..., examples=[{"query": "…"}]),
                                    ident: ConsoleIdentity = _Session):
    """Retrieve, then stream a CRAG answer over Server-Sent Events (text/event-stream).

    Retrieval and relevance grading complete first (a `sources` frame carries the
    matched chunks); only the final generation streams token by token. Same
    ownership rules as the non-streaming /query endpoint.
    """
    import json
    from starlette.responses import StreamingResponse
    folder, token = _owner(folder_id, ident)
    q = (body.get("query") or "").strip()
    if len(q) < 2:
        raise HTTPException(422, "query too short (min 2 chars)")

    from src.api.router.rag_query import query_rag
    from src.api.router.response import QueryRequest
    from fastapi import Request as _Req
    req = QueryRequest(
        query=q, folder_name=folder.name,
        top_k=body.get("top_k"), similarity_cutoff=body.get("similarity_cutoff"),
        auto_merge=body.get("auto_merge"))
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
