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

        class UndefinedTable(Exception):
            pgcode = "42P01"

        searcher = MagicMock()
        searcher.search = AsyncMock(side_effect=sqlalchemy.exc.ProgrammingError(
            "s", {}, UndefinedTable('relation "data_1" does not exist')))
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
             patch(f"{_CL}.afetch_rows_in_ranges", new=AsyncMock(return_value={
                 "f1": [(6, "six", None), (7, "seven", None), (8, "eight", None)]})) as rows:
            out = await retr.aquery("q", SearchSettings(top_k=4))
        assert [r.node_role for r in out] == ["parent", "expanded", "expanded"]
        assert out[0].merged_from_leaves == ["a", "b"] and out[0].score == 0.9
        assert out[1].text == "six\n\nseven\n\neight" and out[1].chunk_index_range == [6, 8]
        assert out[2].text == "text d"
        assert rows.await_args.args[1] == [("f1", 6, 8)]           # one batched fetch

    async def test_empty_search_returns_empty(self):
        from src.domain.rag.auto_merging import AutoMergingRetriever

        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=SearchOutcome(nodes=[]))
        retr = AutoMergingRetriever(MagicMock(), searcher=searcher)
        assert await retr.aquery("q", SearchSettings()) == []


class TestKnownParents:
    """plan_merges(known_parents=...) - parent rows read while the reranker ran."""

    async def test_covering_known_parents_replace_both_queries(self):
        leaves = [nws("a", 0.9, parent="P"), nws("b", 0.8, parent="P"), nws("c", 0.7)]
        known = {"P": parent("P", chunk_count=3), "Q": parent("Q", chunk_count=2)}
        with patch(f"{_CL}.aget_parent_children_counts", new=AsyncMock()) as counts, \
             patch(f"{_CL}.afetch_nodes", new=AsyncMock()) as fetch:
            groups = await mp.plan_merges(MagicMock(), leaves, 0.5, known_parents=known)
        counts.assert_not_awaited()
        fetch.assert_not_awaited()
        assert list(groups) == ["P"] and groups["P"].parent is known["P"]

    async def test_known_parents_follow_the_same_rule(self):
        leaves = [nws("a", 0.9, parent="P"), nws("b", 0.8, parent="P")]
        below = {"P": parent("P", chunk_count=5)}                 # 2/5 < 0.5
        no_count = {"P": parent("P")}                              # no chunk_count: no merge
        not_parent = {"P": TextNode(id_="P", text="x", metadata={"chunk_count": 2})}
        for known in (below, no_count, not_parent):
            assert await mp.plan_merges(MagicMock(), leaves, 0.5, known_parents=known) == {}
        assert await mp.plan_merges(MagicMock(), leaves, 1.1, known_parents=below) == {}

    async def test_missing_parent_falls_back_to_the_queries(self):
        leaves = [nws("a", 0.9, parent="P"), nws("b", 0.8, parent="P")]
        with patch(f"{_CL}.aget_parent_children_counts",
                   new=AsyncMock(return_value={"P": 2})) as counts, \
             patch(f"{_CL}.afetch_nodes", new=AsyncMock(return_value={"P": parent("P")})):
            groups = await mp.plan_merges(MagicMock(), leaves, 0.5,
                                          known_parents={"Q": parent("Q", chunk_count=2)})
        counts.assert_awaited_once()
        assert list(groups) == ["P"]


class TestRetrieverPrefetch:
    """The MCP retriever reads merge / neighbour rows while the reranker runs."""

    def _leaves(self):
        return [nws("a", parent="P", chunk_index=0), nws("b", parent="P", chunk_index=1),
                nws("c", chunk_index=7), nws("z", parent="Z", chunk_index=20)]

    def _retriever(self, reranker, n=1, threshold=0.5):
        from src.domain.rag.auto_merging import AutoMergingRetriever

        s = searcher_with(self._leaves(), [], reranker=reranker)
        return AutoMergingRetriever(MagicMock(), merge_threshold=threshold,
                                    expand_context_neighbors=n, searcher=s)

    def _rows(self):
        return {"f1": [(0, "zero", "P"), (1, "one", "P"), (2, "two", "P"),
                       (6, "six", None), (7, "seven", None), (8, "eight", None),
                       (19, "n19", "Z"), (20, "n20", "Z"), (21, "n21", "Z")]}

    async def test_prefetch_replaces_the_post_rerank_queries(self):
        from tests.test_hybrid_search import FakeReranker

        retr = self._retriever(FakeReranker({"a": 0.9, "b": 0.8, "c": 0.7, "z": 0.1}))
        with patch(f"{_CL}.aget_parent_children_counts", new=AsyncMock()) as counts, \
             patch(f"{_CL}.afetch_nodes",
                   new=AsyncMock(return_value={"P": parent("P", chunk_count=2)})) as fetch, \
             patch(f"{_CL}.afetch_rows_in_ranges",
                   new=AsyncMock(return_value=self._rows())) as rows:
            out = await retr.aquery("q", SearchSettings(top_k=3))
        counts.assert_not_awaited()
        assert fetch.await_count == 1 and fetch.await_args.args[1] == ["P"]
        assert rows.await_count == 1                     # every pool leaf, z included
        assert ("f1", 19, 21) in rows.await_args.args[1]
        assert [r.node_role for r in out] == ["parent", "expanded"]
        assert out[1].text == "six\n\nseven\n\neight"
        assert "prefetch" in retr.last_outcome.timings

    async def test_same_results_as_without_prefetch(self):
        from tests.test_hybrid_search import FakeReranker
        from src.domain.rag.auto_merging import AutoMergingRetriever

        table = {"a": 0.9, "b": 0.8, "c": 0.7, "z": 0.1}
        patches = dict(
            aget_parent_children_counts=AsyncMock(return_value={"P": 2}),
            afetch_nodes=AsyncMock(return_value={"P": parent("P", chunk_count=2)}),
            afetch_rows_in_ranges=AsyncMock(return_value=self._rows()))
        with patch.multiple(_CL, **patches):
            with_pre = await self._retriever(FakeReranker(table)).aquery(
                "q", SearchSettings(top_k=3))
            leaves = [n for n in self._leaves() if n.node.node_id != "z"]
            for n, sc in zip(leaves, (0.9, 0.8, 0.7)):
                n.score = sc
            searcher = MagicMock()
            searcher.search = AsyncMock(return_value=SearchOutcome(nodes=leaves))
            plain = await AutoMergingRetriever(
                MagicMock(), merge_threshold=0.5, expand_context_neighbors=1,
                searcher=searcher).aquery("q", SearchSettings(top_k=3))
        assert [(r.node_id, r.text, r.node_role, r.chunk_index_range) for r in with_pre] == \
               [(r.node_id, r.text, r.node_role, r.chunk_index_range) for r in plain]

    async def test_empty_prefetch_answers_are_redone(self):
        """The lookups return {} on a DB error: that must not mean "no merge"."""
        from tests.test_hybrid_search import FakeReranker

        retr = self._retriever(FakeReranker({"a": 0.9, "b": 0.8, "c": 0.7}))
        with patch(f"{_CL}.aget_parent_children_counts",
                   new=AsyncMock(return_value={"P": 2})) as counts, \
             patch(f"{_CL}.afetch_nodes",
                   new=AsyncMock(side_effect=[{}, {"P": parent("P")}])), \
             patch(f"{_CL}.afetch_rows_in_ranges",
                   new=AsyncMock(side_effect=[{}, self._rows()])) as rows:
            out = await retr.aquery("q", SearchSettings(top_k=3))
        counts.assert_awaited_once()
        assert rows.await_count == 2
        assert [r.node_role for r in out] == ["parent", "expanded"]

    async def test_nothing_to_prefetch(self):
        from tests.test_hybrid_search import FakeReranker

        # merging off and no expansion: no prefetch at all
        retr = self._retriever(FakeReranker({"a": 1.0}), n=0, threshold=1.1)
        with patch(f"{_CL}.afetch_nodes", new=AsyncMock()) as fetch, \
             patch(f"{_CL}.afetch_rows_in_ranges", new=AsyncMock()) as rows:
            out = await retr.aquery("q", SearchSettings(top_k=1))
        fetch.assert_not_awaited()
        rows.assert_not_awaited()
        assert retr.last_outcome.prefetched is None and out[0].node_role == "leaf"

    async def test_prefetch_without_merge_candidates_reads_rows_only(self):
        from src.domain.rag.auto_merging import AutoMergingRetriever
        from tests.test_hybrid_search import FakeReranker

        s = searcher_with([nws("c", chunk_index=7)], [], reranker=FakeReranker({"c": 1.0}))
        retr = AutoMergingRetriever(MagicMock(), merge_threshold=0.5,
                                    expand_context_neighbors=1, searcher=s)
        with patch(f"{_CL}.afetch_nodes", new=AsyncMock()) as fetch, \
             patch(f"{_CL}.afetch_rows_in_ranges",
                   new=AsyncMock(return_value=self._rows())) as rows:
            out = await retr.aquery("q", SearchSettings(top_k=1))
        fetch.assert_not_awaited()
        assert rows.await_count == 1 and out[0].text == "six\n\nseven\n\neight"
