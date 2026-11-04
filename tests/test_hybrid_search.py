"""The shared retrieval path (src/domain/rag/hybrid_search.py) and its users.

REST, MCP, the console and the eval harness must retrieve identically; these
tests pin the fusion rules, the per-request settings (no shared mutable state)
and that the MCP retriever goes through the same searcher as REST.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.domain.rag import hybrid_search as hs
from src.domain.rag.hybrid_search import (
    HybridSearcher,
    SearchSettings,
    concat_fuse,
    rrf_fuse,
)


def nws(nid, score=0.5, role="leaf", parent=None, text=None, **md):
    meta = dict(md)
    if role is not None:
        meta["node_role"] = role
    if parent:
        meta["parent_node_id"] = parent
    meta.setdefault("file_id", "f1")
    meta.setdefault("file_name", "a.txt")
    return NodeWithScore(node=TextNode(id_=nid, text=text or f"text {nid}", metadata=meta),
                         score=score)


def ids(nodes):
    return [n.node.node_id for n in nodes]


class FakeReranker:
    """Scores by a fixed table; mimics Reranker.arerank (top_n, in-place scores)."""

    def __init__(self, table=None, fail=False):
        self.table = table or {}
        self.fail = fail
        self.calls = []

    async def arerank(self, query, nodes, top_n=None):
        self.calls.append((query, ids(nodes), top_n))
        scored = sorted(nodes, key=lambda n: self.table.get(n.node.node_id, 0.0), reverse=True)
        for n in scored:
            n.score = self.table.get(n.node.node_id, 0.0)
        return scored[:top_n] if top_n else scored


def searcher_with(dense, sparse, reranker=None):
    s = HybridSearcher(MagicMock(), reranker=reranker, index=MagicMock())
    # Fresh NodeWithScore wrappers per call (fusion / rerank rewrite .score in place).
    s.dense = AsyncMock(side_effect=lambda q, k: [NodeWithScore(node=n.node, score=n.score)
                                                  for n in dense][:k])
    s.sparse = AsyncMock(side_effect=lambda q, k, st: [NodeWithScore(node=n.node, score=n.score)
                                                       for n in sparse][:k])
    return s


# --------------------------------------------------------------- settings

class TestSearchSettings:
    def _cfg(self, **ret):
        base = dict(default_top_k=7, default_similarity_cutoff=0.3, default_sparse_top_k=20,
                    hybrid_fusion="concat", rrf_k=30, text_search_config="english")
        base.update(ret)
        return SimpleNamespace(rag=SimpleNamespace(retrieval=SimpleNamespace(**base)))

    def test_reads_config_and_applies_overrides(self):
        with patch("src.config.config_manager.Config.get_config_model", return_value=self._cfg()):
            st = SearchSettings.from_config(top_k=3, similarity_cutoff=None)
        assert st == SearchSettings(top_k=3, similarity_cutoff=0.3, sparse_top_k=20,
                                    fusion="concat", rrf_k=30, text_search_config="english")

    def test_missing_fields_fall_back_to_defaults(self):
        cfg = SimpleNamespace(rag=SimpleNamespace(retrieval=SimpleNamespace(default_top_k=4)))
        with patch("src.config.config_manager.Config.get_config_model", return_value=cfg):
            st = SearchSettings.from_config()
        assert st.top_k == 4 and st.fusion == "rrf" and st.sparse_top_k == 12

    def test_unreadable_config_uses_defaults(self):
        with patch("src.config.config_manager.Config.get_config_model",
                   side_effect=RuntimeError("no config")):
            assert SearchSettings.from_config() == SearchSettings()

    def test_unknown_fusion_becomes_rrf(self):
        assert SearchSettings(fusion="hybrid").fusion == "rrf"

    def test_settings_are_immutable(self):
        with pytest.raises(Exception):
            SearchSettings().top_k = 9  # type: ignore[misc]


# ----------------------------------------------------------------- fusion

class TestFusion:
    def test_rrf_rewards_agreement_and_truncates(self):
        dense = [nws("a"), nws("b"), nws("c")]
        sparse = [nws("c"), nws("d")]
        fused = rrf_fuse([dense, sparse], k=3, rrf_k=60)
        assert ids(fused)[0] == "c"            # in both lists
        assert len(fused) == 3
        assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)

    def test_concat_is_dense_then_bm25_only_untruncated(self):
        dense = [nws("a", 0.9), nws("b", 0.2)]
        sparse = [nws("b", 3.0), nws("x", 2.0), nws("y", 1.0)]
        assert ids(concat_fuse(dense, sparse)) == ["a", "b", "x", "y"]

    def test_concat_cutoff_only_filters_dense(self):
        dense = [nws("a", 0.9), nws("b", 0.2)]
        sparse = [nws("x", 0.01)]
        assert ids(concat_fuse(dense, sparse, cutoff=0.5)) == ["a", "x"]


# ----------------------------------------------------------------- search

class TestSearch:
    async def test_rrf_without_reranker_returns_top_k(self):
        s = searcher_with([nws("a"), nws("b"), nws("c")], [nws("c"), nws("a")])
        out = await s.search("q", SearchSettings(top_k=2))
        assert ids(out.nodes) == ["a", "c"]
        assert not out.reranked and out.candidates == 2

    async def test_reranker_gets_at_least_three_times_top_k_candidates(self):
        dense = [nws(c) for c in "abcdefgh"]
        rr = FakeReranker({"f": 0.9, "e": 0.8, "h": 1.0})
        s = searcher_with(dense, [], reranker=rr)
        # rerank_candidates=1 isolates the 3 × top_k floor (the default pool is 60)
        out = await s.search("q", SearchSettings(top_k=2, rerank_candidates=1,
                                                 rerank_pool="fused"))
        assert s.dense.await_args.args[1] == 6          # max(3 × top_k, rerank_candidates)
        assert ids(out.nodes) == ["f", "e"] and out.reranked  # h was never a candidate
        assert rr.calls[0][2] == 2

    async def test_concat_union_goes_to_reranker(self):
        rr = FakeReranker({"x": 0.99})
        s = searcher_with([nws("a", 0.9)], [nws("x", 1.0)], reranker=rr)
        out = await s.search("q", SearchSettings(top_k=1, fusion="concat"))
        assert rr.calls[0][1] == ["a", "x"] and ids(out.nodes) == ["x"]

    async def test_concurrent_requests_keep_their_own_settings(self):
        """One cached searcher serves concurrent requests with different top_k."""
        dense = [nws(c) for c in "abcdefghij"]
        s = searcher_with(dense, [])
        orig = s.dense.side_effect

        async def slow_dense(q, k):
            await asyncio.sleep(0.01 if k == 1 else 0)
            return orig(q, k)

        s.dense = AsyncMock(side_effect=slow_dense)
        small, big = await asyncio.gather(
            s.search("q", SearchSettings(top_k=1)), s.search("q", SearchSettings(top_k=8)))
        assert len(small.nodes) == 1 and len(big.nodes) == 8

    async def test_trace_lists_keep_raw_scores(self):
        s = searcher_with([nws("a", 0.83)], [nws("a", 0.41)])
        out = await s.search("q", SearchSettings(top_k=1))
        assert out.dense[0].score == pytest.approx(0.83)
        assert out.sparse[0].score == pytest.approx(0.41)
        assert out.nodes[0].score == pytest.approx(2 / 61)       # RRF


# ------------------------------------------------------- dense / sparse

class _DenseConn:
    """Async connection stub for the dense query: records statements, returns rows."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt, params=None):
        self.calls.append((str(stmt), dict(params or {})))
        if "SET LOCAL" in str(stmt):
            return SimpleNamespace(fetchall=lambda: [])
        k = (params or {}).get("k", len(self.rows))
        return SimpleNamespace(fetchall=lambda: self.rows[:k])


def fake_dense_engine(rows):
    """(engine, conn): ``engine.begin()`` yields a transaction-scoped connection stub."""
    conn = _DenseConn(rows)
    engine = MagicMock()
    engine.begin.return_value = conn
    return engine, conn


def _dense_row(nid, sim, role="leaf"):
    meta = {"file_id": "f1"}
    if role is not None:
        meta["node_role"] = role
    return (nid, f"text {nid}", meta, sim)


class TestDense:
    def _searcher(self, rows):
        s = HybridSearcher(MagicMock(table_name="data_1_x"), index=MagicMock())
        s.embed_query = AsyncMock(return_value=[0.5, -0.25])
        engine, conn = fake_dense_engine(rows)
        s._dense_engine = MagicMock(return_value=engine)
        return s, conn

    async def test_dense_keeps_leaves_only(self):
        s, _ = self._searcher([_dense_row("a", 0.9), _dense_row("p", 0.8, role="parent"),
                               _dense_row("old", 0.7, role=None)])
        out = await s.dense("q", 5)
        assert ids(out) == ["a", "old"]
        assert out[0].score == pytest.approx(0.9)

    async def test_ef_search_is_raised_transaction_locally_to_k(self):
        s, conn = self._searcher([_dense_row(str(i), 0.5) for i in range(60)])
        out = await s.dense("q", 60)
        assert len(out) == 60                                   # not cut at 40
        set_sql, _ = conn.calls[0]
        assert set_sql == "SET LOCAL hnsw.ef_search = 60"       # LOCAL: never leaks
        sql, params = conn.calls[1]
        assert params["k"] == 60 and params["emb"] == "[0.5,-0.25]"
        assert '"data_1_x"' in sql and "<=>" in sql

    async def test_ef_search_never_below_pgvector_default(self):
        s, conn = self._searcher([_dense_row("a", 0.5)])
        await s.dense("q", 3)
        assert conn.calls[0][0] == f"SET LOCAL hnsw.ef_search = {hs.DEFAULT_EF_SEARCH}"

    async def test_embed_query_uses_the_index_embed_model(self):
        index = MagicMock()
        index._embed_model.aget_query_embedding = AsyncMock(return_value=[1.0])
        s = HybridSearcher(MagicMock(), index=index)
        assert await s.embed_query("hello") == [1.0]
        index._embed_model.aget_query_embedding.assert_awaited_once_with("hello")

    def test_dense_engine_falls_back_to_shared_engine(self):
        s = HybridSearcher(MagicMock(), index=MagicMock())      # MagicMock is no AsyncEngine
        with patch("db.db.get_async_engine", return_value="shared"):
            assert s._dense_engine() == "shared"

    def test_dense_engine_prefers_llama_index_engine(self):
        from sqlalchemy.ext.asyncio import AsyncEngine

        eng = MagicMock(spec=AsyncEngine)
        s = HybridSearcher(MagicMock(_async_engine=eng), index=MagicMock())
        assert s._dense_engine() is eng


class TestScoreTypeAndTimings:
    def test_score_type(self):
        assert hs.SearchOutcome(nodes=[], reranked=True).score_type == "rerank"
        assert hs.SearchOutcome(nodes=[]).score_type == "rrf"
        assert hs.SearchOutcome(nodes=[], fusion="concat").score_type == "vector"
        # a failed rerank leaves the fusion scores in place
        assert hs.SearchOutcome(nodes=[], rerank_failed=True).score_type == "rrf"

    async def test_search_records_stage_timings(self):
        rr = FakeReranker({"a": 0.9})
        s = searcher_with([nws("a")], [nws("a")], reranker=rr)
        out = await s.search("q", SearchSettings(top_k=1))
        assert {"fuse", "rerank"} <= set(out.timings)
        assert all(v >= 0 for v in out.timings.values())
        assert out.score_type == "rerank"

    def test_record_stage_outside_a_search_is_a_noop(self):
        hs.record_stage("dense", 0.0)                           # must not raise

    async def test_concurrent_searches_keep_separate_timings(self):
        s = searcher_with([nws("a")], [])
        a, b = await asyncio.gather(s.search("q", SearchSettings(top_k=1)),
                                    s.search("q", SearchSettings(top_k=1)))
        assert a.timings is not b.timings


# ------------------------------------------------- MCP uses the same path

class TestMcpUsesSharedPath:
    async def test_auto_merging_retriever_segments_and_fuses(self):
        """The MCP retriever must not use llama-index's native hybrid on the raw query."""
        from src.domain.rag.auto_merging import AutoMergingRetriever

        from tests.test_bm25_search import _row, _searcher_with_db

        seg = MagicMock()
        seg.segment.return_value = "退貨 政策"
        searcher, conn, engine = _searcher_with_db({"'退貨' | '政策'": [_row("b", 2.0)]})
        del searcher.segment_query                              # use the real one
        searcher.dense = AsyncMock(return_value=[nws("a", 0.8)])
        retr = AutoMergingRetriever(MagicMock(), merge_threshold=1.1,
                                    expand_context_neighbors=0, searcher=searcher)
        with patch("src.domain.rag.text_segmenter.get_segmenter", return_value=seg), \
             patch("db.db.get_async_engine", return_value=engine):
            out = await retr.aquery("退貨政策", SearchSettings(top_k=5), expand_context=False)
        assert searcher.dense.await_args.args[0] == "退貨政策"
        bm25 = [p for sql, p in conn.calls if "q_any" in p][0]
        assert bm25["q_all"] == "'退貨' & '政策'"                 # segmented BM25
        assert {r.node_id for r in out} == {"a", "b"}          # BM25 hit reaches agents
        assert retr.last_outcome is not None
        assert {"merge"} <= set(retr.last_outcome.timings)

    async def test_keyword_args_still_accepted(self):
        from src.domain.rag.auto_merging import AutoMergingRetriever

        searcher = searcher_with([nws("a")], [])
        retr = AutoMergingRetriever(MagicMock(), merge_threshold=1.1,
                                    expand_context_neighbors=0, searcher=searcher)
        with patch("src.config.config_manager.Config.get_config_model", side_effect=RuntimeError):
            out = await retr.aquery(query_text="q", top_k=3, similarity_cutoff=0.0,
                                    expand_context=False)
        assert [r.node_id for r in out] == ["a"] and out[0].node_role == "leaf"


def test_rerank_candidate_factor_is_three():
    assert hs.RERANK_CANDIDATE_FACTOR == 3


class TestRerankCandidatePool:
    """The reranker's pool must not shrink with top_k (cross-lingual hits sit at fused rank 30-60)."""

    def _searcher(self, reranker):
        return HybridSearcher(MagicMock(), reranker=reranker, index=MagicMock())

    def test_pool_is_at_least_rerank_candidates(self):
        s = self._searcher(MagicMock())
        assert s.candidate_depth(SearchSettings(top_k=10)) == 60       # was 30 = 3 x top_k
        assert s.candidate_depth(SearchSettings(top_k=5, rerank_candidates=40)) == 40

    def test_large_top_k_still_gets_three_times(self):
        s = self._searcher(MagicMock())
        assert s.candidate_depth(SearchSettings(top_k=30, rerank_candidates=60)) == 90

    def test_without_reranker_depth_is_top_k(self):
        s = self._searcher(None)
        assert s.candidate_depth(SearchSettings(top_k=10, rerank_candidates=60)) == 10

    def test_read_from_config_and_part_of_the_cache_key(self):
        from types import SimpleNamespace

        cfg = SimpleNamespace(rag=SimpleNamespace(retrieval=SimpleNamespace(
            default_top_k=10, default_similarity_cutoff=0.0, rerank_candidates=80)))
        with patch("src.config.config_manager.Config.get_config_model", return_value=cfg):
            st = SearchSettings.from_config()
        assert st.rerank_candidates == 80
        a = hs.cache_fingerprint("q", SearchSettings(rerank_candidates=60))
        b = hs.cache_fingerprint("q", SearchSettings(rerank_candidates=80))
        assert a != b


class TestPerListRerankPool:
    """rerank_pool=per_list: dense top N + BM25 top M go to the cross-encoder."""

    def _searcher(self, reranker):
        return HybridSearcher(MagicMock(), reranker=reranker, index=MagicMock())

    def test_defaults(self):
        st = SearchSettings()
        assert st.rerank_pool == "per_list"
        assert (st.rerank_dense_candidates, st.rerank_sparse_candidates) == (30, 10)

    def test_unknown_pool_becomes_per_list(self):
        assert SearchSettings(rerank_pool="bogus").rerank_pool == "per_list"

    def test_depths_keep_three_times_top_k(self):
        assert hs.per_list_depths(SearchSettings(top_k=10)) == (30, 10)
        assert hs.per_list_depths(SearchSettings(top_k=20)) == (60, 10)
        assert hs.per_list_depths(SearchSettings(top_k=1, rerank_dense_candidates=0,
                                                 rerank_sparse_candidates=-3)) == (3, 0)

    def test_retrieval_depth_covers_both_pools(self):
        s = self._searcher(MagicMock())
        # The fused list (trace view, rerank-failure fallback) keeps its old depth...
        assert s.candidate_depth(SearchSettings(top_k=10)) == 60
        # ...and grows when a per-list depth is deeper than it.
        assert s.candidate_depth(SearchSettings(top_k=10, rerank_dense_candidates=90)) == 90
        assert s.candidate_depth(SearchSettings(top_k=10, rerank_sparse_candidates=70)) == 70
        assert s.candidate_depth(SearchSettings(top_k=10, rerank_dense_candidates=90,
                                                rerank_pool="fused")) == 60

    def test_pool_is_dense_head_plus_bm25_head_in_fused_order(self):
        s = self._searcher(MagicMock())
        dense = [nws(c) for c in "abcdef"]
        sparse = [nws(c) for c in "xyaz"]
        fused = rrf_fuse([dense, sparse], k=10)          # a first (in both lists)
        st = SearchSettings(top_k=1, rerank_dense_candidates=3, rerank_sparse_candidates=2)
        pool = s.rerank_pool(fused, dense, sparse, st)
        assert sorted(ids(pool)) == ["a", "b", "c", "x", "y"]
        assert ids(pool) == [i for i in ids(fused) if i in {"a", "b", "c", "x", "y"}]

    def test_pool_reaches_past_the_fused_cut(self):
        """A dense-only hit below the fused cut still reaches the reranker."""
        s = self._searcher(MagicMock())
        dense = [nws(c) for c in "abcd"]
        sparse = [nws(c) for c in "xyz"]
        fused = rrf_fuse([dense, sparse], k=3)            # a, x, b
        st = SearchSettings(top_k=1, rerank_dense_candidates=4, rerank_sparse_candidates=1)
        pool = s.rerank_pool(fused, dense, sparse, st)
        assert ids(pool) == ["a", "x", "b", "c", "d"]

    def test_fused_mode_and_concat_fusion_keep_the_fused_list(self):
        s = self._searcher(MagicMock())
        dense, sparse = [nws(c) for c in "abc"], [nws(c) for c in "xy"]
        fused = rrf_fuse([dense, sparse], k=5)
        tiny = dict(rerank_dense_candidates=1, rerank_sparse_candidates=0, top_k=1)
        assert ids(s.rerank_pool(fused, dense, sparse,
                                 SearchSettings(rerank_pool="fused", **tiny))) == ids(fused)
        union = concat_fuse(dense, sparse)
        assert ids(s.rerank_pool(union, dense, sparse,
                                 SearchSettings(fusion="concat", **tiny))) == ids(union)

    async def test_search_reranks_the_per_list_pool(self):
        dense = [nws(c) for c in "abcdefgh"]
        sparse = [nws(c) for c in "xyzw"]
        rr = FakeReranker({"h": 1.0, "w": 0.9, "f": 0.8})
        s = searcher_with(dense, sparse, reranker=rr)
        st = SearchSettings(top_k=2, rerank_candidates=1, rerank_dense_candidates=6,
                            rerank_sparse_candidates=2)
        out = await s.search("q", st)
        assert sorted(rr.calls[0][1]) == sorted("abcdefxy")   # h, w are beyond the heads
        assert ids(out.nodes) == ["f", "a"]
        assert out.candidates == 8 and out.reranked
        assert len(out.fused) == 6           # depth max(3 × top_k, 1, 6, 2); x, y lie beyond it

    async def test_failed_rerank_falls_back_to_fused_order(self):
        dense = [nws(c) for c in "abcd"]
        sparse = [nws(c) for c in "xa"]
        s = searcher_with(dense, sparse, reranker=_FailingScorer())
        out = await s.search("q", SearchSettings(top_k=2, rerank_dense_candidates=1,
                                                 rerank_sparse_candidates=1))
        assert out.rerank_failed and ids(out.nodes) == ["a", "x"]


class _FailingScorer:
    score_threshold = 0.0

    async def ascore(self, query, texts):
        raise RuntimeError("reranker down")


class TestRerankPrefetch:
    """search(prefetch=...) overlaps side work with the rerank call."""

    async def test_prefetch_gets_the_pool_and_lands_on_the_outcome(self):
        seen = []

        async def prefetch(pool):
            seen.append(ids(pool))
            return {"rows": len(pool)}

        rr = FakeReranker({"b": 1.0})
        s = searcher_with([nws(c) for c in "abc"], [], reranker=rr)
        out = await s.search("q", SearchSettings(top_k=1), prefetch=prefetch)
        assert seen == [rr.calls[0][1]]
        assert out.prefetched == {"rows": 3} and ids(out.nodes) == ["b"]

    async def test_prefetch_runs_concurrently_with_rerank(self):
        order = []

        class SlowReranker(FakeReranker):
            async def arerank(self, query, nodes, top_n=None):
                order.append("rerank start")
                await asyncio.sleep(0.02)
                order.append("rerank end")
                return await super().arerank(query, nodes, top_n)

        async def prefetch(pool):
            order.append("prefetch")
            return True

        s = searcher_with([nws("a")], [], reranker=SlowReranker())
        await s.search("q", SearchSettings(top_k=1), prefetch=prefetch)
        assert order.index("prefetch") < order.index("rerank end")

    async def test_failing_prefetch_is_ignored(self):
        async def prefetch(pool):
            raise RuntimeError("db down")

        s = searcher_with([nws("a")], [], reranker=FakeReranker({"a": 1.0}))
        out = await s.search("q", SearchSettings(top_k=1), prefetch=prefetch)
        assert out.prefetched is None and ids(out.nodes) == ["a"]

    async def test_no_prefetch_result_when_rerank_failed_or_absent(self):
        async def prefetch(pool):
            return "data"

        s = searcher_with([nws("a")], [], reranker=_FailingScorer())
        out = await s.search("q", SearchSettings(top_k=1), prefetch=prefetch)
        assert out.rerank_failed and out.prefetched is None
        s = searcher_with([nws("a")], [])
        out = await s.search("q", SearchSettings(top_k=1), prefetch=prefetch)
        assert out.prefetched is None

    async def test_cancelled_rerank_cancels_the_prefetch(self):
        started = asyncio.Event()
        holder = {}

        class HangingReranker(FakeReranker):
            async def arerank(self, query, nodes, top_n=None):
                started.set()
                await asyncio.sleep(10)

        async def prefetch(pool):
            holder["task"] = asyncio.current_task()
            await asyncio.sleep(10)

        s = searcher_with([nws("a")], [], reranker=HangingReranker())
        task = asyncio.ensure_future(s.search("q", SearchSettings(top_k=1), prefetch=prefetch))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert holder["task"].cancelled() or holder["task"].done()
