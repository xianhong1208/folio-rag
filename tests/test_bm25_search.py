"""BM25 (Postgres full-text search) on the shared retrieval path.

- leaves only, filtered in SQL (parents' long text used to win ts_rank and fill
  the LIMIT before a Python-side filter threw them away);
- AND of all query terms first, OR only to fill up to k;
- question / function words dropped;
- depth = max(k, sparse_top_k) — sparse_top_k used to be ignored by llama-index;
- dense and BM25 run concurrently.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


from src.domain.rag import hybrid_search as hs
from src.domain.rag.hybrid_search import HybridSearcher, SearchSettings
from tests.test_hybrid_search import nws


class TestLexemes:
    def test_drops_question_and_function_words(self):
        assert hs.query_lexemes("退貨 政策 是 什麼 ?") == ["退貨", "政策"]
        assert hs.query_lexemes("What is the refund policy ?") == ["refund", "policy"]

    def test_keeps_everything_when_only_stop_words(self):
        assert hs.query_lexemes("what is it") == ["what", "is", "it"]

    def test_punctuation_dropped_digits_kept_deduped(self):
        assert hs.query_lexemes("第 3 條 , 第 3 條 !") == ["第", "3", "條"]
        assert hs.query_lexemes("") == []

    def test_tsquery_quotes_terms(self):
        assert hs.to_tsquery_text(["a", "it's", "x\\y"], "&") == "'a' & 'it''s' & 'x\\\\y'"


class _Conn:
    """Async connection stub: returns canned rows per query string, records calls."""

    def __init__(self, rows_by_query):
        self.rows_by_query = rows_by_query
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt, params):
        self.calls.append((str(stmt), dict(params)))
        rows = self.rows_by_query.get(params["q"], [])
        return SimpleNamespace(fetchall=lambda: rows[: params["k"]])


def _row(nid, rank, role="leaf"):
    return (nid, f"text {nid}", {"node_role": role, "file_id": "f"}, rank)


def _searcher_with_db(rows_by_query, segmented="退貨 政策 是 什麼"):
    conn = _Conn(rows_by_query)
    engine = MagicMock()
    engine.connect.return_value = conn
    s = HybridSearcher(MagicMock(table_name="data_1_x"), index=MagicMock())
    s.segment_query = AsyncMock(return_value=segmented)
    return s, conn, engine


class TestSparseSql:
    async def test_and_first_then_or_backfill_leaf_only_in_sql(self):
        rows = {"'退貨' & '政策'": [_row("a", 0.9)],
                "'退貨' | '政策'": [_row("a", 0.9), _row("b", 0.5), _row("c", 0.4)]}
        s, conn, engine = _searcher_with_db(rows)
        with patch("db.db.get_async_engine", return_value=engine):
            out = await s.sparse("退貨政策是什麼", 3, SearchSettings())
        assert [n.node.node_id for n in out] == ["a", "b", "c"]
        sql0, p0 = conn.calls[0]
        assert p0["q"] == "'退貨' & '政策'" and p0["cfg"] == "simple" and p0["k"] == 3
        assert conn.calls[1][1]["q"] == "'退貨' | '政策'"
        assert "IS DISTINCT FROM 'parent'" in sql0 and '"data_1_x"' in sql0
        assert out[0].score == 0.9 and out[0].node.metadata["file_id"] == "f"

    async def test_enough_and_hits_skip_the_or_query(self):
        rows = {"'退貨' & '政策'": [_row("a", 0.9), _row("b", 0.8)]}
        s, conn, engine = _searcher_with_db(rows)
        with patch("db.db.get_async_engine", return_value=engine):
            out = await s.sparse("q", 2, SearchSettings())
        assert len(conn.calls) == 1 and len(out) == 2

    async def test_single_term_runs_one_query(self):
        rows = {"'退貨'": [_row("a", 0.9)]}
        s, conn, engine = _searcher_with_db(rows, segmented="退貨")
        with patch("db.db.get_async_engine", return_value=engine):
            out = await s.sparse("q", 5, SearchSettings())
        assert len(conn.calls) == 1 and [n.node.node_id for n in out] == ["a"]

    async def test_no_terms_no_query(self):
        s, conn, engine = _searcher_with_db({}, segmented="? !")
        with patch("db.db.get_async_engine", return_value=engine):
            assert await s.sparse("?!", 5, SearchSettings()) == []
        assert conn.calls == []

    async def test_db_failure_degrades_to_dense_only(self):
        s, _, _ = _searcher_with_db({})
        with patch("db.db.get_async_engine", side_effect=RuntimeError("down")):
            assert await s.sparse("q", 5, SearchSettings()) == []

    async def test_legacy_rows_without_node_content_still_load(self):
        rows = {"'退貨'": [("a", "raw text", None, 0.3)]}
        s, _, engine = _searcher_with_db(rows, segmented="退貨")
        with patch("db.db.get_async_engine", return_value=engine):
            out = await s.sparse("q", 5, SearchSettings())
        assert out[0].node.get_content() == "raw text"


class TestDepthAndConcurrency:
    async def test_sparse_depth_honours_sparse_top_k(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        s.dense = AsyncMock(return_value=[])
        s.sparse = AsyncMock(return_value=[])
        await s.candidates("q", SearchSettings(top_k=5, sparse_top_k=40))
        assert s.sparse.await_args.args[1] == 40
        await s.candidates("q", SearchSettings(top_k=50, sparse_top_k=4))
        assert s.sparse.await_args.args[1] == 50

    async def test_dense_and_sparse_run_concurrently(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        started = []

        async def dense(q, k):
            started.append("dense")
            await asyncio.sleep(0.05)
            return [nws("a")]

        async def sparse(q, k, st):
            started.append("sparse")
            await asyncio.sleep(0.05)
            return [nws("b")]

        s.dense, s.sparse = dense, sparse
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        fused, _, _ = await s.candidates("q", SearchSettings(top_k=2))
        assert loop.time() - t0 < 0.09
        assert {n.node.node_id for n in fused} == {"a", "b"}


class TestSegmentQuery:
    async def test_non_simple_config_uses_raw_words(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        assert await s.segment_query("refund policies", SearchSettings(
            text_search_config="english")) == "refund policies"

    async def test_simple_config_segments_in_python(self):
        seg = MagicMock()
        seg.segment.return_value = "退貨 政策"
        s = HybridSearcher(MagicMock(), index=MagicMock())
        with patch("src.domain.rag.ckip_segmenter.get_segmenter", return_value=seg):
            assert await s.segment_query("退貨政策", SearchSettings()) == "退貨 政策"

    async def test_segmentation_failure_falls_back_to_raw(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        with patch("src.domain.rag.ckip_segmenter.get_segmenter", side_effect=RuntimeError):
            assert await s.segment_query("raw text", SearchSettings()) == "raw text"
