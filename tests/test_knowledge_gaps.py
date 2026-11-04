"""Knowledge gaps: judged on the reranker's top score, latest record per question,
with MCP searches logged too.

A question is a gap when its most recent search (per folder) returned nothing,
or its best rerank score was below rag.rerank.gap_threshold. Asking again after
documents were added clears it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.db import QueryLog
from db.query_log_db import QueryLogDB


@pytest.fixture()
def db():
    engine = create_engine("sqlite://")
    QueryLog.__table__.create(engine)
    S = sessionmaker(bind=engine)
    with patch("db.query_log_db.Session", S):
        yield S


def _add(S, query, folder=1, results=3, top=None, ago_min=10):
    with S() as s:
        s.add(QueryLog(query=query, folder_id=folder, result_count=results, latency_ms=50,
                       top_score=top, token_prefix="t…",
                       created_at=datetime.now(timezone.utc) - timedelta(minutes=ago_min)))
        s.commit()


class TestRecord:
    def test_top_score_is_stored(self, db):
        QueryLogDB.record(query="q", folder_id=1, result_count=2, latency_ms=10,
                          token_prefix=None, top_score=0.42)
        with db() as s:
            assert s.query(QueryLog.top_score).scalar() == pytest.approx(0.42)

    def test_top_score_is_optional(self, db):
        QueryLogDB.record(query="q", folder_id=1, result_count=0, latency_ms=10, token_prefix=None)
        with db() as s:
            assert s.query(QueryLog.top_score).scalar() is None


class TestAnalytics:
    def test_low_top_score_is_a_gap_even_with_results(self, db):
        _add(db, "weak match", results=5, top=0.2)
        _add(db, "good match", results=5, top=0.9)
        _add(db, "unreranked", results=5, top=None)
        _add(db, "nothing", results=0, top=None)
        out = QueryLogDB.analytics(days=30, gap_threshold=0.5)
        gaps = {g["query"]: g for g in out["gap_queries"]}
        assert set(gaps) == {"weak match", "nothing"}
        assert gaps["weak match"]["top_score"] == pytest.approx(0.2)
        assert out["gap_count"] == 2 and out["gap_threshold"] == 0.5
        assert out["zero_result"] == 1                       # unchanged KPI

    def test_latest_record_per_question_decides(self, db):
        _add(db, "fixed later", results=0, ago_min=60)
        _add(db, "fixed later", results=0, ago_min=50)
        _add(db, "fixed later", results=4, top=0.95, ago_min=5)   # docs were added
        _add(db, "broke later", results=4, top=0.95, ago_min=60)
        _add(db, "broke later", results=4, top=0.1, ago_min=5)
        out = QueryLogDB.analytics(days=30, gap_threshold=0.5)
        gaps = {g["query"]: g for g in out["gap_queries"]}
        assert set(gaps) == {"broke later"}
        assert gaps["broke later"]["count"] == 2

    def test_same_question_in_two_folders_is_judged_per_folder(self, db):
        _add(db, "q", folder=1, results=0, ago_min=5)
        _add(db, "q", folder=2, results=3, top=0.9, ago_min=1)
        out = QueryLogDB.analytics(days=30, gap_threshold=0.5)
        assert [(g["query"], g["folder_id"]) for g in out["gap_queries"]] == [("q", 1)]
        assert QueryLogDB.analytics(days=30, folder_id=2, gap_threshold=0.5)["gap_queries"] == []

    def test_gap_threshold_defaults_to_config(self, db):
        _add(db, "weak", results=2, top=0.6)
        cfg = SimpleNamespace(rag=SimpleNamespace(rerank=SimpleNamespace(gap_threshold=0.7)))
        with patch("src.config.config_manager.Config.get_config_model", return_value=cfg):
            assert [g["query"] for g in QueryLogDB.analytics(days=30)["gap_queries"]] == ["weak"]

    def test_empty_window(self, db):
        out = QueryLogDB.analytics(days=30, gap_threshold=0.5)
        assert out["gap_queries"] == [] and out["gap_count"] == 0


class TestConfig:
    def test_gap_threshold_setting(self):
        from src.config.model import RerankConfig
        from src.config.runtime_overrides import EDITABLE
        assert RerankConfig().gap_threshold == 0.5
        assert EDITABLE["rag.rerank.gap_threshold"] == "live"


class TestMcpLogging:
    async def test_mcp_search_is_logged_with_top_score(self):
        from src.adapter.rag_query import RAGQueryService
        from src.domain.rag.hybrid_search import SearchOutcome

        ctx = MagicMock()
        ctx.merge_threshold, ctx.auto_merging_enabled, ctx.expand_neighbors = 0.5, True, 0
        svc = RAGQueryService(ctx)
        folder = SimpleNamespace(id=4, name="docs", vector_table_uuid="u")
        retriever = MagicMock()
        retriever.last_outcome = SearchOutcome(nodes=[], top_score=0.31)

        async def fake_search(**kw):
            return {"mode": "search", "total_results": 2, "results": [{}, {}]}

        with patch("src.domain.rag.folder_acl.verify_folder_access", return_value=folder), \
             patch("src.adapter.rag_query._query_semaphore", asyncio.Semaphore(4)), \
             patch("src.adapter.rag_query.AutoMergingRetriever", return_value=retriever), \
             patch("src.domain.rag.agentic_handlers.handle_search", new=AsyncMock(side_effect=fake_search)), \
             patch("db.query_log_db.QueryLogDB.record") as rec:
            out = await svc.query_agentic(folder_name="docs", mode="search", query="what?",
                                          file_id=None, top_k=5, similarity_cutoff=0.0,
                                          expand_context=False, token="tok-123456789")
            from src.adapter.rag_query import flush_query_logs

            await flush_query_logs()   # the row is written after the response
        assert out["total_results"] == 2
        kw = rec.call_args.kwargs
        assert kw["query"] == "what?" and kw["folder_id"] == 4 and kw["result_count"] == 2
        assert kw["top_score"] == pytest.approx(0.31) and kw["latency_ms"] >= 0
        assert "tok-123456789" not in (kw["token_prefix"] or "")

    async def test_logging_failure_never_breaks_search(self):
        from src.adapter.rag_query import RAGQueryService

        ctx = MagicMock()
        ctx.merge_threshold, ctx.auto_merging_enabled, ctx.expand_neighbors = 0.5, True, 0
        svc = RAGQueryService(ctx)
        folder = SimpleNamespace(id=4, name="docs", vector_table_uuid="u")
        with patch("src.domain.rag.folder_acl.verify_folder_access", return_value=folder), \
             patch("src.adapter.rag_query._query_semaphore", asyncio.Semaphore(4)), \
             patch("src.adapter.rag_query.AutoMergingRetriever", return_value=MagicMock(last_outcome=None)), \
             patch("src.domain.rag.agentic_handlers.handle_search",
                   new=AsyncMock(return_value={"mode": "search", "results": []})), \
             patch("db.query_log_db.QueryLogDB.record", side_effect=RuntimeError("db down")):
            out = await svc.query_agentic(folder_name="docs", mode="search", query="q?",
                                          file_id=None, top_k=5, similarity_cutoff=0.0,
                                          expand_context=False, token="t")
            from src.adapter.rag_query import flush_query_logs

            await flush_query_logs()
        assert out["mode"] == "search"

    async def test_search_does_not_wait_for_the_log_row(self):
        from src.adapter import rag_query

        ctx = MagicMock()
        ctx.merge_threshold, ctx.auto_merging_enabled, ctx.expand_neighbors = 0.5, True, 0
        svc = rag_query.RAGQueryService(ctx)
        folder = SimpleNamespace(id=4, name="docs", vector_table_uuid="u")
        gate = asyncio.Event()

        async def slow_log(**kw):
            await gate.wait()

        with patch("src.domain.rag.folder_acl.verify_folder_access", return_value=folder), \
             patch("src.adapter.rag_query._query_semaphore", asyncio.Semaphore(4)), \
             patch("src.adapter.rag_query.AutoMergingRetriever", return_value=MagicMock(last_outcome=None)), \
             patch("src.domain.rag.agentic_handlers.handle_search",
                   new=AsyncMock(return_value={"mode": "search", "results": []})), \
             patch("src.adapter.rag_query._log_query", new=slow_log):
            out = await svc.query_agentic(folder_name="docs", mode="search", query="q?",
                                          file_id=None, top_k=5, similarity_cutoff=0.0,
                                          expand_context=False, token="t")
            assert out["mode"] == "search" and len(rag_query._pending_logs) == 1
            gate.set()
            await rag_query.flush_query_logs()
        assert not rag_query._pending_logs
