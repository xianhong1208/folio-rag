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

    async def test_reranker_gets_three_times_top_k_candidates(self):
        dense = [nws(c) for c in "abcdefgh"]
        rr = FakeReranker({"f": 0.9, "e": 0.8, "h": 1.0})
        s = searcher_with(dense, [], reranker=rr)
        out = await s.search("q", SearchSettings(top_k=2))
        assert s.dense.await_args.args[1] == 6          # 3 × top_k
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

class _Recorder:
    """Stands in for VectorIndexRetriever; records (mode, query)."""

    def __init__(self, hits_by_mode):
        self.hits_by_mode = hits_by_mode
        self.calls = []

    def __call__(self, index=None, similarity_top_k=None, vector_store_query_mode=None, **kw):
        rec = self

        class _R:
            async def aretrieve(self_inner, q):
                rec.calls.append((vector_store_query_mode, q, similarity_top_k, kw))
                return list(rec.hits_by_mode.get(vector_store_query_mode, []))

        return _R()


class TestDense:
    async def test_dense_keeps_leaves_only(self):
        rec = _Recorder({"default": [nws("a"), nws("p", role="parent"), nws("old", role=None)]})
        s = HybridSearcher(MagicMock(), index=MagicMock())
        with patch("llama_index.core.retrievers.VectorIndexRetriever", rec):
            assert ids(await s.dense("q", 5)) == ["a", "old"]
        assert rec.calls[0][0] == "default"


# ------------------------------------------------- MCP uses the same path

class TestMcpUsesSharedPath:
    async def test_auto_merging_retriever_segments_and_fuses(self):
        """The MCP retriever must not use llama-index's native hybrid on the raw query."""
        from src.domain.rag.auto_merging import AutoMergingRetriever

        from tests.test_bm25_search import _row, _searcher_with_db

        rec = _Recorder({"default": [nws("a", 0.8)]})
        seg = MagicMock()
        seg.segment.return_value = "退貨 政策"
        searcher, conn, engine = _searcher_with_db({"'退貨' & '政策'": [_row("b", 2.0)]})
        del searcher.segment_query                              # use the real one
        retr = AutoMergingRetriever(MagicMock(), merge_threshold=1.1,
                                    expand_context_neighbors=0, searcher=searcher)
        with patch("llama_index.core.retrievers.VectorIndexRetriever", rec), \
             patch("src.domain.rag.ckip_segmenter.get_segmenter", return_value=seg), \
             patch("db.db.get_async_engine", return_value=engine):
            out = await retr.aquery("退貨政策", SearchSettings(top_k=5), expand_context=False)
        assert [c[0] for c in rec.calls] == ["default"]         # no native "hybrid"
        assert conn.calls[0][1]["q"] == "'退貨' & '政策'"         # segmented BM25
        assert {r.node_id for r in out} == {"a", "b"}          # BM25 hit reaches agents
        assert retr.last_outcome is not None

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
