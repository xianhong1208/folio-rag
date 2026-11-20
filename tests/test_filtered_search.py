"""Filtered search inside one folder: the document filter's file ids reach the dense and
BM25 SQL, the dense query turns on pgvector's iterative scan, the cache key includes
the filter, and an edit of tags / metadata drops the folder's cached results."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.domain.rag import hybrid_search as hs
from src.domain.rag.doc_filter import DocumentFilter
from src.domain.rag.hybrid_search import HybridSearcher, SearchSettings, cache_fingerprint


class _Conn:
    def __init__(self, rows, fail_iterative=False):
        self.rows, self.calls, self.fail_iterative = rows, [], fail_iterative

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def begin_nested(self):
        return self

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        self.calls.append((sql, dict(params or {})))
        if sql == hs.ITERATIVE_SCAN_SQL and self.fail_iterative:
            raise RuntimeError('unrecognized configuration parameter "hnsw.iterative_scan"')
        if sql.startswith("SET LOCAL"):
            return SimpleNamespace(fetchall=lambda: [])
        return SimpleNamespace(fetchall=lambda: self.rows[: params.get("k", len(self.rows))])


def _row(nid, sim, fid="f1"):
    return (nid, f"text {nid}", {"file_id": fid, "node_role": "leaf"}, sim)


def _dense_searcher(rows, **kw):
    s = HybridSearcher(MagicMock(table_name="data_1_x"), index=MagicMock())
    s.embed_query = AsyncMock(return_value=[0.5])
    conn = _Conn(rows, **kw)
    engine = MagicMock()
    engine.begin.return_value = conn
    s._dense_engine = MagicMock(return_value=engine)
    return s, conn


class TestStatements:
    def test_unfiltered_statements_unchanged(self):
        assert "file_ids" not in str(hs._dense_statement("t"))
        assert "file_ids" not in str(hs._bm25_statement("t"))

    def test_filtered_statements_bind_file_ids(self):
        dense = str(hs._dense_statement("t", True))
        bm25 = str(hs._bm25_statement("t", True))
        for sql in (dense, bm25):
            assert "(metadata_->>'file_id') = ANY(CAST(:file_ids AS text[]))" in sql
        assert dense.index("WHERE") < dense.index("ORDER BY")


class TestDenseFiltered:
    async def test_iterative_scan_on_and_ids_bound(self):
        s, conn = _dense_searcher([_row("a", 0.4), _row("b", 0.9)])
        out = await s.dense("q", 5, file_ids=("f1", "f2"))
        sqls = [c[0] for c in conn.calls]
        assert sqls[0].startswith("SET LOCAL hnsw.ef_search")
        assert sqls[1] == "SET LOCAL hnsw.iterative_scan = relaxed_order"
        assert conn.calls[2][1]["file_ids"] == ["f1", "f2"]
        # relaxed_order may return rows out of order: re-sorted by similarity
        assert [n.node.node_id for n in out] == ["b", "a"]

    async def test_unfiltered_dense_has_no_iterative_scan(self):
        s, conn = _dense_searcher([_row("a", 0.4)])
        await s.dense("q", 5)
        assert hs.ITERATIVE_SCAN_SQL not in [c[0] for c in conn.calls]
        assert "file_ids" not in conn.calls[-1][1]

    async def test_old_pgvector_still_answers(self):
        hs._ITERATIVE_SCAN_WARNED = False
        s, conn = _dense_searcher([_row("a", 0.4)], fail_iterative=True)
        out = await s.dense("q", 5, file_ids=("f1",))
        assert [n.node.node_id for n in out] == ["a"] and hs._ITERATIVE_SCAN_WARNED

    async def test_no_matching_file_means_no_query(self):
        s, conn = _dense_searcher([_row("a", 0.4)])
        assert await s.dense("q", 5, file_ids=()) == [] and conn.calls == []


class TestSparseFiltered:
    async def test_bm25_binds_file_ids(self):
        s = HybridSearcher(MagicMock(table_name="data_1_x"), index=MagicMock())
        s.folder_segmenter = AsyncMock(return_value=None)
        s.segment_query = AsyncMock(return_value="alpha")
        conn = _Conn([_row("a", 0.3)])
        engine = MagicMock()
        engine.connect.return_value = conn
        with patch("db.db.get_async_engine", return_value=engine):
            out = await s.sparse("alpha", 5, SearchSettings(), file_ids=("f9",))
        sql, params = conn.calls[0]
        assert ":file_ids" in sql and params["file_ids"] == ["f9"]
        assert [n.node.node_id for n in out] == ["a"]

    async def test_empty_ids_skip_the_query(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        assert await s.sparse("q", 5, SearchSettings(), file_ids=()) == []


class TestRetrieveRoutesIds:
    async def test_settings_file_ids_reach_both_paths(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        s.dense = AsyncMock(return_value=[])
        s.sparse = AsyncMock(return_value=[])
        await s.search("q", SearchSettings(file_ids=["b", "a"]))
        assert s.dense.await_args.kwargs == {"file_ids": ("a", "b")}
        assert s.sparse.await_args.kwargs == {"file_ids": ("a", "b")}

    async def test_unfiltered_calls_unchanged(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())
        s.dense = AsyncMock(return_value=[])
        s.sparse = AsyncMock(return_value=[])
        await s.search("q", SearchSettings())
        assert s.dense.await_args.kwargs == {} and s.sparse.await_args.kwargs == {}


class TestCacheKey:
    def _fp(self, **kw):
        with patch("src.domain.rag.text_segmenter.configured_segmenter_name", return_value="ckip"):
            return cache_fingerprint("q", SearchSettings(file_ids=kw.pop("ids", None)),
                                     merge_threshold=0.5, **kw)

    def test_filtered_differs_from_unfiltered(self):
        assert self._fp(ids=("a",), filter_key="f") != self._fp()

    def test_different_matches_differ(self):
        assert self._fp(ids=("a",), filter_key="f") != self._fp(ids=("b",), filter_key="f")

    def test_different_filters_differ(self):
        assert self._fp(ids=("a",), filter_key="f1") != self._fp(ids=("a",), filter_key="f2")


def _ctx(folder_owner="sub:me"):
    ctx = MagicMock()
    ctx.indexing_service.get_folder.return_value = SimpleNamespace(id=4, name="f",
                                                                   user_token=folder_owner)
    ctx.reranker = None
    return ctx


class TestQueryService:
    async def test_no_match_returns_empty_without_searching(self):
        from src.adapter.rag_query import RAGQueryService
        from src.infrastructure.cache.cache_service import CacheService
        CacheService.reset_instance()
        ctx = _ctx()
        svc = RAGQueryService(ctx)
        with patch("src.adapter.rag_query.resolve_file_ids", return_value=()) as res:
            out = await svc.query_rag("q", 4, "sub:me",
                                      filters=DocumentFilter(tags_any=["nothing"]))
        assert out.total_results == 0 and out.results == [] and not out.not_indexed
        assert res.call_args.kwargs == {"owner": "sub:me", "folder_id": 4}
        ctx.get_vector_store.assert_not_called()

    async def test_matches_go_into_the_settings_and_the_cache_key(self):
        from src.adapter.rag_query import RAGQueryService
        from src.infrastructure.cache.cache_service import CacheService
        CacheService.reset_instance()
        ctx = _ctx()
        svc = RAGQueryService(ctx)
        engine = MagicMock()
        outcome = SimpleNamespace(top_score=None, not_indexed=False, rerank_failed=False)
        engine.asearch = AsyncMock(return_value=([], outcome))
        svc._engine_for = MagicMock(return_value=engine)
        flt = DocumentFilter(tags_any=["x"])
        with patch("src.adapter.rag_query.resolve_file_ids", return_value=("id1",)), \
             patch("src.adapter.rag_query.cache_fingerprint", return_value="fp") as fp:
            await svc.query_rag("q", 4, "sub:me", filters=flt)
        settings = engine.asearch.await_args.args[1]
        assert settings.file_ids == ("id1",)
        assert fp.call_args.kwargs["filter_key"] == flt.fingerprint()

    async def test_trace_gets_the_filter(self):
        from src.adapter.rag_query import RAGQueryService
        ctx = _ctx()
        svc = RAGQueryService(ctx)
        engine = MagicMock()
        engine.atrace = AsyncMock(return_value={"results": []})
        svc._engine_for = MagicMock(return_value=engine)
        with patch("src.adapter.rag_query.resolve_file_ids", return_value=("id1",)):
            await svc.query_trace("q", 4, "sub:me", filters=DocumentFilter(tags_any=["x"]))
        assert engine.atrace.await_args.args[1].file_ids == ("id1",)


class TestEditInvalidatesCache:
    async def test_tag_edit_drops_the_folders_query_cache(self):
        from src.adapter.file import FileAdapter
        from src.infrastructure.cache.cache_service import CacheKeys, CacheService
        CacheService.reset_instance()
        cache = CacheService.get_instance()
        cache.set(CacheKeys.rag_query(4, "x"), "stale")
        cache.set(CacheKeys.rag_query(5, "x"), "other folder")
        row = SimpleNamespace(id="00000000-0000-0000-0000-000000000001", folder_id=4, file_name="a", file_path="p", file_size=1,
                              mime_type=None, description=None, tags=["A"], doc_metadata={"y": 1},
                              upload_time=datetime(2024, 1, 1),
                              updated_time=datetime(2024, 1, 1))
        with patch.object(FileAdapter, "_verify_folder_ownership",
                          return_value=SimpleNamespace(id=4)), \
             patch("db.filedb.FileDB.get", return_value=[row]), \
             patch("db.filedb.FileDB.update", return_value=row) as upd:
            out = await FileAdapter.update_file_metadata(4, "u", "sub:me", tags=[" A ", "a"],
                                                         metadata={"y": 1})
        assert upd.call_args.kwargs == {"tags": ["A"], "doc_metadata": {"y": 1}}
        assert out[0].metadata == {"y": 1}
        assert cache.get(CacheKeys.rag_query(4, "x")) is None
        assert cache.get(CacheKeys.rag_query(5, "x")) == "other folder"

    async def test_description_edit_keeps_the_cache(self):
        from src.adapter.file import FileAdapter
        from src.infrastructure.cache.cache_service import CacheKeys, CacheService
        CacheService.reset_instance()
        cache = CacheService.get_instance()
        cache.set(CacheKeys.rag_query(4, "x"), "kept")
        row = SimpleNamespace(id="00000000-0000-0000-0000-000000000001", folder_id=4, file_name="a", file_path="p", file_size=1,
                              mime_type=None, description="d", tags=None,
                              upload_time=datetime(2024, 1, 1), updated_time=datetime(2024, 1, 1))
        with patch.object(FileAdapter, "_verify_folder_ownership",
                          return_value=SimpleNamespace(id=4)), \
             patch("db.filedb.FileDB.get", return_value=[row]), \
             patch("db.filedb.FileDB.update", return_value=row):
            out = await FileAdapter.update_file_metadata(4, "u", "sub:me", description="d")
        assert out[0].tags == [] and out[0].metadata == {}
        assert cache.get(CacheKeys.rag_query(4, "x")) == "kept"
