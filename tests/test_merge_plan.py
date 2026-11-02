"""The shared auto-merge rule and the REST QueryEngine / MCP retriever built on it."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from llama_index.core.schema import TextNode

from src.domain.rag import merge_plan as mp
from src.domain.rag.hybrid_search import SearchOutcome, SearchSettings
from tests.test_hybrid_search import nws, searcher_with

_CL = "src.domain.rag.chunk_lookup.ChunkLookup"


def parent(pid, **md):
    meta = {"node_role": "parent", "file_id": "f1", "file_name": "a.txt",
            "children_index_range": [0, 3], **md}
    return TextNode(id_=pid, text=f"whole section {pid}", metadata=meta)


class TestRule:
    def test_group_by_parent_skips_legacy_leaves(self):
        groups = mp.group_by_parent([nws("a", parent="P"), nws("b"), nws("c", parent="P")])
        assert list(groups) == ["P"] and len(groups["P"]) == 2

    def test_select_needs_two_hits_and_ratio(self):
        hits = {"P": [1, 2], "Q": [1], "R": [1, 2]}
        assert mp.select_parents(hits, {"P": 4, "Q": 1, "R": 5}, 0.5) == ["P"]
        assert mp.select_parents(hits, {"P": 0}, 0.1) == []

    async def test_disabled_threshold_never_queries(self):
        with patch(f"{_CL}.aget_parent_children_counts", new=AsyncMock()) as counts:
            assert await mp.plan_merges(MagicMock(), [nws("a", parent="P"),
                                                      nws("b", parent="P")], 1.1) == {}
            assert await mp.plan_merges(MagicMock(), [], None) == {}
        counts.assert_not_awaited()

    async def test_single_hits_skip_the_count_query(self):
        with patch(f"{_CL}.aget_parent_children_counts", new=AsyncMock()) as counts:
            assert await mp.plan_merges(MagicMock(), [nws("a", parent="P")], 0.1) == {}
        counts.assert_not_awaited()

    async def test_below_ratio_fetches_no_parents(self):
        with patch(f"{_CL}.aget_parent_children_counts", new=AsyncMock(return_value={"P": 10})), \
             patch(f"{_CL}.afetch_nodes", new=AsyncMock()) as fetch:
            out = await mp.plan_merges(MagicMock(), [nws("a", parent="P"),
                                                     nws("b", parent="P")], 0.5)
        assert out == {} and not fetch.await_count

    async def test_plan_and_arrange(self):
        leaves = [nws("a", 0.9, parent="P"), nws("x", 0.8), nws("b", 0.7, parent="P")]
        with patch(f"{_CL}.aget_parent_children_counts", new=AsyncMock(return_value={"P": 3})), \
             patch(f"{_CL}.afetch_nodes", new=AsyncMock(return_value={"P": parent("P")})):
            groups = await mp.plan_merges(MagicMock(), leaves, 0.5)
        assert list(groups) == ["P"] and groups["P"].score == 0.9
        assert groups["P"].parent_id == "P"
        arranged = mp.arrange(leaves, groups)
        assert isinstance(arranged[0], mp.MergeGroup) and arranged[1].node.node_id == "x"
        assert len(arranged) == 2


class TestQueryEngine:
    def _engine(self, dense, sparse, reranker=None):
        from src.domain.rag.query_engine import QueryEngine
        return QueryEngine(MagicMock(), searcher=searcher_with(dense, sparse, reranker))

    async def test_parent_nodes_filtered_and_leaf_metadata_mapped(self):
        eng = self._engine([nws("00000000-0000-0000-0000-000000000001", 0.9, page_no=3,
                                headings=["H"], content_type="table")], [])
        out = await eng.aquery("q", SearchSettings(top_k=3))
        md = out[0].metadata
        assert md.page == 3 and md.headings == ["H"] and md.content_type == "table"
        assert md.node_role == "leaf" and md.merged_from is None

    async def test_rest_applies_the_shared_merge_rule(self):
        a = "00000000-0000-0000-0000-00000000000a"
        b = "00000000-0000-0000-0000-00000000000b"
        p = "00000000-0000-0000-0000-0000000000ff"
        eng = self._engine([nws(a, parent=p, page_no=2), nws(b, parent=p)], [])
        with patch(f"{_CL}.aget_parent_children_counts", new=AsyncMock(return_value={p: 2})), \
             patch(f"{_CL}.afetch_nodes", new=AsyncMock(return_value={p: parent(p)})):
            out = await eng.aquery("q", SearchSettings(top_k=5), merge_threshold=0.5)
        assert len(out) == 1
        assert out[0].metadata.node_role == "parent" and out[0].metadata.merged_from == 2
        assert out[0].text.startswith("whole section") and out[0].metadata.page == 2

    async def test_missing_table_becomes_folder_not_found(self):
        import sqlalchemy.exc
        from src.domain.exceptions import FolderNotFoundError
        from src.domain.rag.query_engine import QueryEngine

        searcher = MagicMock()
        searcher.search = AsyncMock(side_effect=sqlalchemy.exc.ProgrammingError(
            "s", {}, Exception('relation "data_1" does not exist')))
        eng = QueryEngine(MagicMock(), searcher=searcher)
        try:
            await eng.aquery("q", SearchSettings())
        except FolderNotFoundError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected FolderNotFoundError")

    async def test_trace_reports_per_path_ranks(self):
        a = "00000000-0000-0000-0000-00000000000a"
        b = "00000000-0000-0000-0000-00000000000b"
        eng = self._engine([nws(a, 0.8), nws(b, 0.7)], [nws(b, 2.5)])
        tr = await eng.atrace("q", SearchSettings(top_k=2))
        assert tr["fusion"] == "rrf" and tr["candidates"] == 2
        row_b = next(r for r in tr["results"] if r["metadata"]["node_id"] == b)
        assert row_b["vector_rank"] == 2 and row_b["bm25_rank"] == 1
        assert row_b["bm25_score"] == 2.5 and row_b["hybrid_rank"] == 1


class TestAutoMergingRetriever:
    async def test_merge_and_expand(self):
        from src.domain.rag.auto_merging import AutoMergingRetriever

        leaves = [nws("a", 0.9, parent="P", chunk_index=0),
                  nws("b", 0.8, parent="P", chunk_index=1),
                  nws("c", 0.7, chunk_index=7),
                  nws("d", 0.6)]                                   # no chunk_index
        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=SearchOutcome(nodes=leaves))
        retr = AutoMergingRetriever(MagicMock(), merge_threshold=0.5,
                                    expand_context_neighbors=1, searcher=searcher)
        with patch(f"{_CL}.aget_parent_children_counts", new=AsyncMock(return_value={"P": 2})), \
             patch(f"{_CL}.afetch_nodes", new=AsyncMock(return_value={"P": parent("P")})), \
             patch(f"{_CL}.afetch_neighbor_windows",
                   new=AsyncMock(return_value=[{"text": "c-window", "index_range": [6, 8]}])):
            out = await retr.aquery("q", SearchSettings(top_k=4))
        assert [r.node_role for r in out] == ["parent", "expanded", "expanded"]
        assert out[0].merged_from_leaves == ["a", "b"] and out[0].score == 0.9
        assert out[1].text == "c-window" and out[1].chunk_index_range == [6, 8]
        assert out[2].text == "text d"

    async def test_empty_search_returns_empty(self):
        from src.domain.rag.auto_merging import AutoMergingRetriever

        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=SearchOutcome(nodes=[]))
        retr = AutoMergingRetriever(MagicMock(), searcher=searcher)
        assert await retr.aquery("q", SearchSettings()) == []
