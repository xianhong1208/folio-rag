
"""Control Center management API — CRUD and operations for folders / files / index.

Act-as-owner: a folder is bound to a user_token, and every console operation
calls the existing REST endpoint functions with that owner token, so all
security and cleanup logic stays single-source. An ownerless folder returns 409
(ACL fail-closed). Endpoints are unauthenticated, matching the admin module.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from src.log import get_api_logger

logger = get_api_logger()

router = APIRouter(tags=["Admin: Manage"])


def _normalize_owner_key(owner_token: str) -> str:
    """Normalize an admin-supplied owner token to the per-token ownership key.

    The console's "Owner Token" field may hold either a full MCP Center token (a
    JWT, ~900 chars) or a bare owner key already copied from an existing folder's
    owner. Ownership is keyed on the JWT ``jti`` everywhere else (the Bearer path
    runs ``extract_token`` -> ``owner_key_from_bearer``), so a JWT is reduced to its
    ``jti`` here too; anything that is not a JWT is treated as an already-final owner
    key. Without this, the raw token was stored verbatim and overflowed
    ``user_token`` (varchar 256) — this admin path calls ``create_folder`` directly
    as a function, so the ``extract_token`` dependency never runs.
    """
    import jwt

    try:
        claims = jwt.decode(
            owner_token, options={"verify_signature": False, "verify_aud": False}
        )
        jti = claims.get("jti")
        if jti:
            return str(jti)
    except Exception:
        pass
    return owner_token


def _owner(folder_id: int):
    """Load the folder and return (folder, owner_token); refuse to manage an ownerless folder."""
    from db.cached_folderdb import CachedFolderDB
    folder = CachedFolderDB.get_by_id(folder_id)
    if not folder:
        raise HTTPException(404, f"Folder {folder_id} not found")
    if not folder.user_token:
        raise HTTPException(
            409, f"Folder {folder_id} has no owner token — console cannot manage it")
    return folder, folder.user_token


@router.post("/api/admin/manage/folders")
async def admin_create_folder(
    body: dict = Body(..., example={"name": "docs", "description": "", "owner_token": "…"}),
):
    """Create a folder (owner_token required — the folder is bound to an owner who accesses it with that token)."""
    name = (body.get("name") or "").strip()
    owner_token = (body.get("owner_token") or "").strip()
    if not name:
        raise HTTPException(422, "name is required")
    if not owner_token:
        raise HTTPException(422, "owner_token is required — folders are token-scoped")
    from src.api.router.folder_api import create_folder
    from src.api.router.response import CreateFolderRequest
    return await create_folder(
        CreateFolderRequest(name=name, description=body.get("description")),
        user_token=_normalize_owner_key(owner_token),
    )


@router.patch("/api/admin/manage/folders/{folder_id}")
async def admin_update_folder(folder_id: int, body: dict = Body(...)):
    """Rename / change description (only patches the fields provided)."""
    _, token = _owner(folder_id)
    from src.api.router.folder_api import update_folder_api
    from src.api.router.response import UpdateFolderRequest
    return await update_folder_api(
        folder_id,
        UpdateFolderRequest(name=body.get("name"), description=body.get("description")),
        user_token=token,
    )


@router.delete("/api/admin/manage/folders/{folder_id}")
async def admin_delete_folder(folder_id: int):
    """Delete a folder (reuses the existing endpoint: cancel in-flight jobs first, then soft-delete files and vectors)."""
    _, token = _owner(folder_id)
    from src.api.router.folder_api import delete_folder
    return await delete_folder(folder_id, user_token=token)


@router.post("/api/admin/manage/folders/{folder_id}/files")
async def admin_upload_file(
    folder_id: int,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    auto_index: bool = Form(True),
):
    """Upload a file to the folder (auto_index=True runs the existing background indexing flow)."""
    _, token = _owner(folder_id)
    from src.api.router.file_api import upload_file
    return await upload_file(
        folder_id, file=file, description=description, tags=None,
        auto_index=auto_index, user_token=token,
    )


@router.get("/api/admin/manage/folders/{folder_id}/files/{file_id}/download")
async def admin_download_file(folder_id: int, file_id: UUID):
    _, token = _owner(folder_id)
    from src.api.router.file_api import download_file
    return await download_file(folder_id, file_id, user_token=token)


@router.delete("/api/admin/manage/folders/{folder_id}/files/{file_id}")
async def admin_delete_file(folder_id: int, file_id: UUID):
    """Delete a file (the existing endpoint also clears the file's vectors and index records)."""
    _, token = _owner(folder_id)
    from src.api.router.file_api import delete_file
    return await delete_file(folder_id, file_id, user_token=token)


@router.post("/api/admin/manage/folders/{folder_id}/index")
async def admin_index_folder(
    folder_id: int,
    body: dict = Body(default={}, example={"skip_existing": True}),
):
    """Start a background indexing job for the whole folder.

    skip_existing=True is incremental (only files without an index record).
    skip_existing=False routes to the reindex endpoint (delete old records, then
    rerun); a plain index cannot rebuild because the per-file content_hash
    short-circuit skips unchanged files, so settings changes would have no effect.
    """
    folder, token = _owner(folder_id)
    from src.api.router.response import IndexRequest
    skip_existing = bool(body.get("skip_existing", True))
    if skip_existing:
        from src.api.router.rag_indexing import index_folder_endpoint
        return await index_folder_endpoint(
            folder_id, IndexRequest(), skip_existing=True, token=token, folder=folder)
    from src.api.router.rag_indexing import reindex_folder_endpoint
    return await reindex_folder_endpoint(
        folder_id, IndexRequest(), skip_existing=False, token=token, folder=folder)


@router.post("/api/admin/manage/folders/{folder_id}/rebuild-fts")
async def admin_rebuild_fts(folder_id: int):
    """Backfill the folder's CKIP full-text search vector (`text_search_tsv`).

    Folders indexed before the CKIP migration carry llama-index's native
    ``to_tsvector('simple', text)``, which cannot segment space-less Chinese, so
    BM25 recall on Chinese queries is poor. This re-segments every stored chunk
    with CKIP and overwrites the tsvector in place — no re-embedding, no
    re-parsing, only the search vector changes. New indexing already rebuilds it
    (hierarchical_indexer step 6b); this is the one-click backfill for existing
    folders. No-op (409) unless FTS is in CKIP mode.

    Blocking (sync DB + CKIP inference), so it runs in a threadpool off the loop.
    """
    folder, _ = _owner(folder_id)

    # Only meaningful in CKIP mode; other FTS configs keep the native tsvector.
    from src.config.config_manager import Config
    ret = getattr(getattr(Config.get_config_model(), "rag", None), "retrieval", None)
    if (getattr(ret, "text_search_config", "simple") or "simple") != "simple":
        raise HTTPException(
            409, "FTS is not in CKIP mode "
            "(rag.retrieval.text_search_config != 'simple'); nothing to rebuild")

    from starlette.concurrency import run_in_threadpool
    from db.db import get_engine
    from src.domain.rag.vector_store_manager import (
        VectorStoreManager, rebuild_text_search_tsv)
    from src.domain.rag.ckip_segmenter import get_segmenter

    table_name = VectorStoreManager.physical_table_name(
        folder_id, folder.vector_table_uuid)

    def _work() -> int:
        return rebuild_text_search_tsv(get_engine(), table_name, get_segmenter())

    try:
        rows = await run_in_threadpool(_work)
    except Exception as e:
        logger.warning(f"[ADMIN] rebuild-fts failed for folder {folder_id}: {e}")
        raise HTTPException(500, f"Rebuild failed: {e}")
    return {"data": {"folder_id": folder_id, "rows_updated": rows},
            "message": f"Rebuilt full-text search for {rows} chunk(s)"}


@router.post("/api/admin/manage/folders/{folder_id}/files/{file_id}/retry")
async def admin_retry_file(folder_id: int, file_id: UUID):
    """Retry indexing a single file (force bypasses the content_hash short-circuit; one-click rerun for failed files)."""
    _, token = _owner(folder_id)
    from db.filedb import FileDB
    from src.domain.rag.dto import FileRequest
    from src.adapter.rag import get_rag_adapter
    rows = FileDB.get(id=file_id)
    if not rows:
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
async def admin_query_folder(folder_id: int, body: dict = Body(..., example={"query": "…"})):
    """Run one retrieval against the folder (query test bench; act-as-owner). Returns matched chunks + scores.

    trace=true: run the retrieval trace (breaking out vector / BM25 / hybrid /
    rerank ranks) and build results from the trace's final hits (without rerunning
    a normal query).
    """
    folder, token = _owner(folder_id)
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
async def admin_query_folder_stream(folder_id: int, body: dict = Body(..., example={"query": "…"})):
    """Retrieve, then stream a CRAG answer over Server-Sent Events (text/event-stream).

    Retrieval and relevance grading complete first (a `sources` frame carries the
    matched chunks); only the final generation streams token by token. Same
    act-as-owner semantics as the non-streaming /query endpoint.
    """
    import json
    from starlette.responses import StreamingResponse
    folder, token = _owner(folder_id)
    q = (body.get("query") or "").strip()
    if len(q) < 2:
        raise HTTPException(422, "query too short (min 2 chars)")

    from src.api.router.rag_query import query_rag
    from src.api.router.response import QueryRequest
    from fastapi import Request as _Req
    req = QueryRequest(
        query=q, folder_name=folder.name,
        top_k=body.get("top_k"), similarity_cutoff=body.get("similarity_cutoff"))
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
async def admin_cancel_job(folder_id: int, job_id: str):
    """Cancel a running indexing job."""
    folder, token = _owner(folder_id)
    from src.api.router.rag_indexing import cancel_index_job
    return await cancel_index_job(folder_id, job_id, token=token, folder=folder)
