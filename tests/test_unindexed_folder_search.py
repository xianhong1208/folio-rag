"""Searching a folder that was never indexed returns an empty result, not an error.

A folder's pgvector table is created by its first indexing run. Before 1.4.2 an MCP
search on a folder with no table answered ``internal_error`` (``UndefinedTable``) and
REST answered 404 "Vector store table missing". The searcher now treats its own
missing table as an empty folder; any other missing relation still fails.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sqlalchemy.exc

from src.domain.rag.hybrid_search import (
    HybridSearcher,
    SearchOutcome,
    SearchSettings,
    is_missing_table_error,
)

TABLE = "data_7_1f2e"


class _AsyncpgUndefinedTable(Exception):
    """Shaped like asyncpg.exceptions.UndefinedTableError (SQLSTATE in .sqlstate)."""
    sqlstate = "42P01"


class _Psycopg2UndefinedTable(Exception):
    pgcode = "42P01"


class _UndefinedColumn(Exception):
    sqlstate = "42703"


def _wrapped(inner):
    """The way SQLAlchemy's asyncpg dialect reports it: ProgrammingError -> adapter -> driver."""
    adapter = Exception(f"<class 'asyncpg.exceptions.UndefinedTableError'>: {inner}")
    adapter.__cause__ = inner
    return sqlalchemy.exc.ProgrammingError("SELECT ...", {}, adapter)


class TestRecognition:
    def test_asyncpg_chain(self):
        e = _wrapped(_AsyncpgUndefinedTable(f'relation "{TABLE}" does not exist'))
        assert is_missing_table_error(e)
        assert is_missing_table_error(e, TABLE)

    def test_psycopg2_pgcode(self):
        e = sqlalchemy.exc.ProgrammingError("s", {}, _Psycopg2UndefinedTable(
            f'relation "{TABLE}" does not exist'))
        assert is_missing_table_error(e, TABLE)

    def test_only_the_named_table_counts(self):
        e = _wrapped(_AsyncpgUndefinedTable('relation "Files" does not exist'))
        assert not is_missing_table_error(e, TABLE)

    def test_a_missing_column_is_not_a_missing_table(self):
        e = _wrapped(_UndefinedColumn('column "embedding" does not exist'))
        assert not is_missing_table_error(e)

    def test_plain_errors_are_not_missing_tables(self):
        assert not is_missing_table_error(RuntimeError(f'relation "{TABLE}" does not exist'))


def _searcher():
    return HybridSearcher(MagicMock(table_name=TABLE), index=MagicMock())


class TestSearcher:
    async def test_its_own_missing_table_is_an_empty_folder(self):
        s = _searcher()
        err = _wrapped(_AsyncpgUndefinedTable(f'relation "{TABLE}" does not exist'))
        with patch.object(HybridSearcher, "_retrieve", new=AsyncMock(side_effect=err)):
            out = await s.search("q", SearchSettings(top_k=5))
        assert out.nodes == [] and out.not_indexed is True and out.candidates == 0
        assert out.top_score is None and out.reranked is False

    async def test_another_missing_table_still_fails(self):
        s = _searcher()
        err = _wrapped(_AsyncpgUndefinedTable('relation "Files" does not exist'))
        with patch.object(HybridSearcher, "_retrieve", new=AsyncMock(side_effect=err)):
            with pytest.raises(sqlalchemy.exc.ProgrammingError):
                await s.search("q", SearchSettings(top_k=5))

    async def test_other_failures_still_fail(self):
        s = _searcher()
        with patch.object(HybridSearcher, "_retrieve", new=AsyncMock(side_effect=ConnectionError("db"))):
            with pytest.raises(ConnectionError):
                await s.search("q", SearchSettings(top_k=5))

    async def test_a_normal_empty_search_is_not_flagged(self):
        s = _searcher()
        with patch.object(HybridSearcher, "_retrieve", new=AsyncMock(return_value=([], [], [], [], []))):
            out = await s.search("q", SearchSettings(top_k=5))
        assert out.nodes == [] and out.not_indexed is False


class TestRestEngine:
    async def test_rest_gets_an_empty_list(self):
        from src.domain.rag.query_engine import QueryEngine
        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=SearchOutcome(nodes=[], not_indexed=True))
        results, outcome = await QueryEngine(MagicMock(), searcher=searcher).asearch(
            "q", SearchSettings(top_k=5))
        assert results == [] and outcome.not_indexed


class TestMcpResponse:
    async def test_empty_shape_with_a_not_indexed_hint(self):
        from src.domain.rag.agentic_handlers import _HINT_NO_RESULTS, handle_search
        retriever = MagicMock()
        retriever.aquery = AsyncMock(return_value=[])
        retriever.last_outcome = SearchOutcome(nodes=[], not_indexed=True)
        retriever.score_type = "rerank"
        folder = SimpleNamespace(id=7, name="docs")
        out = await handle_search(query="revenue growth", folder=folder, retriever=retriever,
                                  top_k=5, similarity_cutoff=None, expand_context=True)
        assert out["results"] == [] and out["total_results"] == 0
        assert out["low_relevance"] is True and "error" not in out
        assert "nothing indexed yet" in out["_hint"] and out["_hint"] != _HINT_NO_RESULTS

    async def test_an_indexed_folder_with_no_match_keeps_the_old_hint(self):
        from src.domain.rag.agentic_handlers import _HINT_NO_RESULTS, handle_search
        retriever = MagicMock()
        retriever.aquery = AsyncMock(return_value=[])
        retriever.last_outcome = SearchOutcome(nodes=[])
        out = await handle_search(query="revenue growth", folder=SimpleNamespace(id=7, name="docs"),
                                  retriever=retriever, top_k=5, similarity_cutoff=None,
                                  expand_context=True)
        assert out["_hint"] == _HINT_NO_RESULTS

    async def test_it_is_not_logged_as_a_knowledge_gap(self):
        from src.adapter.rag_query import RAGQueryService, flush_query_logs
        ctx = MagicMock()
        ctx.merge_threshold, ctx.auto_merging_enabled, ctx.expand_neighbors = 0.5, True, 0
        folder = SimpleNamespace(id=4, name="docs", vector_table_uuid="u")
        retriever = MagicMock(last_outcome=SearchOutcome(nodes=[], not_indexed=True))
        with patch("src.domain.rag.folder_acl.verify_folder_access", return_value=folder), \
                patch("src.adapter.rag_query._query_semaphore", asyncio.Semaphore(4)), \
                patch("src.adapter.rag_query.AutoMergingRetriever", return_value=retriever), \
                patch("src.domain.rag.agentic_handlers.handle_search",
                      new=AsyncMock(return_value={"mode": "search", "total_results": 0, "results": []})), \
                patch("db.query_log_db.QueryLogDB.record") as rec:
            await RAGQueryService(ctx).query_agentic(
                folder_name="docs", mode="search", query="q?", file_id=None, top_k=5,
                similarity_cutoff=0.0, expand_context=False, token="t")
            await flush_query_logs()
        rec.assert_not_called()


class TestRestAdapter:
    async def test_not_indexed_result_is_flagged_and_never_cached(self):
        from src.adapter.rag_query import RAGQueryService
        from src.domain.rag.query_engine import QueryEngine
        ctx = MagicMock()
        ctx.merge_threshold, ctx.auto_merging_enabled = 0.5, True
        ctx.query_engines = {}
        ctx.indexing_service.get_folder.return_value = SimpleNamespace(id=7, name="empty")
        cache = MagicMock()
        cache.get.return_value = None
        with patch("src.adapter.rag_query._query_semaphore", asyncio.Semaphore(4)), \
                patch("src.adapter.rag_query.CacheService.get_instance", return_value=cache), \
                patch.object(QueryEngine, "asearch",
                             new=AsyncMock(return_value=([], SearchOutcome(nodes=[], not_indexed=True)))), \
                patch.object(QueryEngine, "__init__", return_value=None):
            out = await RAGQueryService(ctx).query_rag(query="q", folder_id=7, token="t")
        assert out.not_indexed is True and out.total_results == 0
        cache.set.assert_not_called()

    def test_rest_endpoint_says_so_and_does_not_log(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from src.adapter.model import RAGQueryResult
        from src.api.dependencies.auth import extract_token
        from src.api.router import rag_query as router_mod
        from src.auth.dependencies import authenticate_request
        app = FastAPI()
        app.include_router(router_mod.router)
        app.dependency_overrides[authenticate_request] = lambda: {}
        app.dependency_overrides[extract_token] = lambda: "sub:t"
        adapter = MagicMock()
        adapter.query_rag = AsyncMock(return_value=RAGQueryResult(
            query="q", results=[], total_results=0, not_indexed=True))
        with patch.object(router_mod, "get_rag_adapter", return_value=adapter), \
                patch("db.cached_folderdb.CachedFolderDB.get_by_name_and_token",
                      return_value=SimpleNamespace(id=7, name="empty")), \
                patch("db.query_log_db.QueryLogDB.record") as rec:
            r = TestClient(app).post("/query", json={"folder_name": "empty", "query": "revenue"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["data"]["results"] == [] and body["data"]["not_indexed"] is True
        assert body["message"].endswith("nothing indexed yet")
        rec.assert_not_called()
