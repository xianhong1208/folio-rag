"""Blocking work reached from async handlers runs off the event loop: the console's
CRAG answer (two LLM calls), the REST query's folder lookup and query log, the MCP
folder ACL, list mode and the read-mode file lookup."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.adapter.rag_query import RAGQueryService


def _recording(result=None):
    """A sync callable that records the thread it ran on."""
    seen = {}

    def fn(*a, **k):
        seen["thread"] = threading.get_ident()
        return result

    return fn, seen


async def test_folder_acl_runs_in_a_worker_thread():
    from src.domain.rag import folder_acl

    fn, seen = _recording(SimpleNamespace(id=1))
    with patch.object(folder_acl, "verify_folder_access", side_effect=fn):
        out = await folder_acl.averify_folder_access(token="t", folder_name="f")
    assert out.id == 1 and seen["thread"] != threading.get_ident()


async def test_list_mode_runs_in_a_worker_thread():
    ctx = MagicMock()
    svc = RAGQueryService(ctx)
    fn, seen = _recording({"mode": "list"})
    with patch("src.domain.rag.folder_acl.verify_folder_access",
               return_value=SimpleNamespace(id=1, name="f", vector_table_uuid="u")), \
         patch("src.domain.rag.agentic_handlers.handle_list", side_effect=fn):
        out = await svc.query_agentic(folder_name="f", mode="list", query="", file_id=None,
                                      top_k=5, similarity_cutoff=None, expand_context=False,
                                      token="t")
    assert out == {"mode": "list"} and seen["thread"] != threading.get_ident()


async def test_read_mode_passes_offset():
    ctx = MagicMock()
    svc = RAGQueryService(ctx)
    with patch("src.domain.rag.folder_acl.verify_folder_access",
               return_value=SimpleNamespace(id=1, name="f", vector_table_uuid="u")), \
         patch("src.adapter.rag_query.Config.get_config_model",
               return_value=SimpleNamespace(rag=SimpleNamespace(retrieval=SimpleNamespace(
                   max_concurrent_queries=4)))), \
         patch("src.domain.rag.agentic_handlers.handle_read",
               new=AsyncMock(return_value={"mode": "read"})) as read:
        await svc.query_agentic(folder_name="f", mode="read", query="", file_id="x", top_k=5,
                                similarity_cutoff=None, expand_context=False, token="t",
                                offset=42)
    assert read.await_args.kwargs["offset"] == 42


async def test_read_mode_file_lookup_runs_in_a_worker_thread():
    from src.domain.rag import agentic_handlers as ah

    fn, seen = _recording({})
    with patch.object(ah.FileDB, "get_by_ids", side_effect=fn):
        out = await ah.handle_read(file_id="f1", folder=SimpleNamespace(id=1, name="f"),
                                   vector_store=None)
    assert "not found" in out["error"] and seen["thread"] != threading.get_ident()


def _app():
    from fastapi import FastAPI

    from src.api.dependencies.auth import extract_token
    from src.api.router import rag_query as rq
    from src.auth.dependencies import authenticate_request

    app = FastAPI()
    app.include_router(rq.router, prefix="/api/rag")
    app.dependency_overrides[extract_token] = lambda: "tok"
    app.dependency_overrides[authenticate_request] = lambda: None
    return app


def test_rest_query_db_calls_run_off_the_event_loop():
    from fastapi.testclient import TestClient

    from src.adapter.model import RAGQueryResult

    in_loop = {}

    def folder_lookup(name, token):
        in_loop["folder"] = _in_running_loop()
        return SimpleNamespace(id=7, name=name)

    def record(**kw):
        in_loop["log"] = _in_running_loop()

    adapter = MagicMock()
    adapter.query_rag = AsyncMock(return_value=RAGQueryResult(
        query="q", results=[], total_results=0))
    with patch("src.api.router.rag_query.CachedFolderDB.get_by_name_and_token",
               side_effect=folder_lookup), \
         patch("db.query_log_db.QueryLogDB.record", side_effect=record), \
         patch("src.api.router.rag_query.get_rag_adapter", return_value=adapter):
        r = TestClient(_app()).post("/api/rag/query",
                                    json={"query": "hello", "folder_name": "docs"})
    assert r.status_code == 200, r.text
    assert in_loop == {"folder": False, "log": False}


def _in_running_loop() -> bool:
    import asyncio

    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def test_console_answer_generation_runs_in_the_threadpool():
    """_generate_answer makes two blocking LLM calls; the endpoint must not run it
    on the event loop."""
    import asyncio

    from src.api.router import admin_manage as am

    seen = {}

    def gen(q, results):
        seen["in_loop"] = _in_running_loop()
        return {"answer": "A", "confidence": "high", "kept": 1, "dropped": 0, "note": "n"}

    folder = SimpleNamespace(id=7, name="docs")
    body = am.PlaygroundQuery(query="what is it", answer=True)
    qresp = {"data": {"results": [{"text": "ctx", "score": 0.9, "metadata": {}}]}}
    with patch.object(am, "_owner", return_value=(folder, "tok")), \
         patch("src.api.router.rag_query.query_rag", new=AsyncMock(return_value=qresp)), \
         patch.object(am, "_generate_answer", side_effect=gen):
        out = asyncio.run(am.admin_query_folder(7, body, ident=None))
    assert out["data"]["answer"] == "A" and seen["in_loop"] is False
