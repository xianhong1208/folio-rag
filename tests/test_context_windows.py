"""Neighbour windows for MCP search: clipped to the section, merged per file,
never repeating a merged parent's chunks (src/domain/rag/context_windows.py),
and the retriever + chunk lookup that use them."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.domain.rag.context_windows import (
    WindowHit,
    clip_window,
    plan_windows,
    requested_range,
)
from src.domain.rag.hybrid_search import SearchOutcome, SearchSettings
from tests.test_hybrid_search import nws


def hit(nid, idx, rank, parent="P1", score=None, fid="f1"):
    return WindowHit(node_id=nid, file_id=fid, chunk_index=idx, parent_id=parent,
                     rank=rank, score=score if score is not None else 1.0 - rank / 10,
                     text=f"t{idx}")


def rows(spec):
    """{idx: parent} → [(idx, text, parent)]"""
    return [(i, f"t{i}", p) for i, p in sorted(spec.items())]


class TestRanges:
    def test_requested_range_never_negative(self):
        assert requested_range(0, 2) == (0, 2)
        assert requested_range(5, 1) == (4, 6)
        assert requested_range(5, -3) == (5, 5)


class TestClip:
    def test_stays_inside_the_parent_section(self):
        r = {i: (f"t{i}", "P1" if i <= 5 else "P2") for i in range(10)}
        assert clip_window(hit("a", 5, 0), r, 2) == (3, 5)        # 6, 7 are P2
        assert clip_window(hit("b", 6, 0, parent="P2"), r, 2) == (6, 8)

    def test_legacy_hit_without_parent_is_not_clipped(self):
        r = {i: (f"t{i}", "P1" if i < 5 else "P2") for i in range(10)}
        assert clip_window(hit("a", 5, 0, parent=None), r, 2) == (3, 7)

    def test_stops_at_gaps_and_blocked_ranges(self):
        r = {i: (f"t{i}", "P1") for i in (3, 4, 5, 7, 8)}
        assert clip_window(hit("a", 5, 0), r, 2) == (3, 5)        # 6 missing
        r = {i: (f"t{i}", "P1") for i in range(10)}
        assert clip_window(hit("a", 5, 0), r, 2, blocked=[[6, 9]]) == (3, 5)


class TestPlan:
    def test_adjacent_hits_merge_into_one_window(self):
        r = {"f1": rows({i: "P1" for i in range(12)})}
        out = plan_windows([hit("a", 4, 0), hit("b", 6, 1)], r, 1)
        assert len(out) == 1
        w = out[0]
        assert (w.start, w.end) == (3, 7)
        assert w.text == "\n\n".join(f"t{i}" for i in range(3, 8))
        assert [h.node_id for h in w.hits] == ["a", "b"] and w.rank == 0
        assert w.score == 1.0

    def test_touching_windows_merge_far_ones_do_not(self):
        r = {"f1": rows({i: "P1" for i in range(30)})}
        out = plan_windows([hit("a", 2, 0), hit("b", 5, 1), hit("c", 20, 2)], r, 1)
        assert [(w.start, w.end) for w in out] == [(1, 6), (19, 21)]

    def test_touching_windows_of_different_sections_stay_apart(self):
        r = {"f1": rows({**{i: "P1" for i in range(6)}, **{i: "P2" for i in range(6, 12)}})}
        out = plan_windows([hit("a", 4, 0, parent="P1"), hit("b", 7, 1, parent="P2")], r, 2)
        assert [(w.start, w.end) for w in out] == [(2, 5), (6, 9)]

    def test_no_text_is_repeated(self):
        r = {"f1": rows({i: "P1" for i in range(30)})}
        hits = [hit(str(i), i, i) for i in (3, 4, 5, 6, 10, 11)]
        out = plan_windows(hits, r, 5)
        chunks = [c for w in out for c in w.text.split("\n\n")]
        assert len(chunks) == len(set(chunks))

    def test_best_score_and_rank_survive_a_merge(self):
        r = {"f1": rows({i: "P1" for i in range(10)})}
        out = plan_windows([hit("low", 3, 4, score=0.2), hit("top", 4, 1, score=0.9)], r, 1)
        assert out[0].best.node_id == "top" and out[0].score == 0.9 and out[0].rank == 1

    def test_files_are_kept_apart_and_sorted_by_rank(self):
        r = {"f1": rows({i: "P1" for i in range(10)}), "f2": rows({i: "Q" for i in range(10)})}
        out = plan_windows([hit("x", 3, 1, fid="f2", parent="Q"), hit("y", 3, 0)], r, 1)
        assert [w.file_id for w in out] == ["f1", "f2"]

    def test_merged_parent_chunks_are_not_repeated(self):
        r = {"f1": rows({**{i: "P1" for i in range(0, 5)}, **{i: "P2" for i in range(5, 10)}})}
        out = plan_windows([hit("a", 5, 1, parent=None)], r, 3, blocked_by_file={"f1": [[0, 4]]})
        assert (out[0].start, out[0].end) == (5, 8)

    def test_missing_rows_fall_back_to_the_hit_text(self):
        out = plan_windows([hit("a", 7, 0)], {}, 2)
        assert out[0].text == "t7" and (out[0].start, out[0].end) == (7, 7)


# ------------------------------------------------------------- the retriever

class TestRetrieverWindows:
    async def _run(self, leaves, fetched, n=1, groups=None):
        from src.domain.rag.auto_merging import AutoMergingRetriever

        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=SearchOutcome(nodes=leaves))
        retr = AutoMergingRetriever(MagicMock(), merge_threshold=0.5,
                                    expand_context_neighbors=n, searcher=searcher)
        with patch("src.domain.rag.auto_merging.plan_merges",
                   new=AsyncMock(return_value=groups or {})), \
             patch("src.domain.rag.chunk_lookup.ChunkLookup.afetch_rows_in_ranges",
                   new=AsyncMock(return_value=fetched)) as fetch:
            out = await retr.aquery("q", SearchSettings(top_k=len(leaves)))
        return out, fetch, retr

    async def test_adjacent_hits_come_back_as_one_result(self):
        leaves = [nws("a", 0.9, parent="P", chunk_index=4),
                  nws("x", 0.8, chunk_index=1, file_id="f2", parent="Q"),
                  nws("b", 0.7, parent="P", chunk_index=5)]
        fetched = {"f1": [(i, f"t{i}", "P") for i in range(3, 7)],
                   "f2": [(i, f"u{i}", "Q") for i in range(0, 3)]}
        out, fetch, retr = await self._run(leaves, fetched)
        assert [r.node_id for r in out] == ["a", "x"]
        assert out[0].text == "t3\n\nt4\n\nt5\n\nt6" and out[0].chunk_index_range == [3, 6]
        assert out[0].merged_from_leaves == ["a", "b"] and out[0].score == 0.9
        assert out[1].merged_from_leaves == []
        assert fetch.await_count == 1

    async def test_window_skips_a_merged_parents_chunks(self):
        from src.domain.rag.merge_plan import MergeGroup
        from llama_index.core.schema import TextNode

        p = TextNode(id_="P", text="section", metadata={
            "node_role": "parent", "file_id": "f1", "children_index_range": [0, 3]})
        leaves = [nws("a", 0.9, parent="P", chunk_index=1),
                  nws("b", 0.8, parent="P", chunk_index=2),
                  nws("c", 0.7, chunk_index=4)]            # legacy: no parent → unclipped
        groups = {"P": MergeGroup(parent=p, hits=leaves[:2])}
        fetched = {"f1": [(i, f"t{i}", None) for i in range(3, 6)]}
        out, _, _ = await self._run(leaves, fetched, groups=groups)
        assert [r.node_role for r in out] == ["parent", "expanded"]
        assert out[1].chunk_index_range == [4, 5]          # chunk 3 belongs to the parent

    async def test_expand_off_keeps_leaves(self):
        from src.domain.rag.auto_merging import AutoMergingRetriever

        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=SearchOutcome(nodes=[nws("a", chunk_index=1)]))
        retr = AutoMergingRetriever(MagicMock(), merge_threshold=1.1,
                                    expand_context_neighbors=1, searcher=searcher)
        out = await retr.aquery("q", SearchSettings(), expand_context=False)
        assert out[0].node_role == "leaf" and out[0].text == "text a"


# ---------------------------------------------------------- the batched fetch

class _AsyncConn:
    def __init__(self, rows=None, fail=False):
        self.rows, self.fail, self.calls = rows or [], fail, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt, params):
        if self.fail:
            raise RuntimeError("db down")
        self.calls.append((str(stmt), dict(params)))
        return SimpleNamespace(fetchall=lambda: self.rows)


class TestFetchRowsInRanges:
    async def test_overlapping_ranges_are_merged_and_rows_grouped(self):
        from src.domain.rag.chunk_lookup import ChunkLookup

        conn = _AsyncConn([("f1", 3, "t3", "P"), ("f1", 4, "t4", "P"), ("f2", 0, "u0", None)])
        engine = MagicMock()
        engine.connect.return_value = conn
        with patch("db.db.get_async_engine", return_value=engine):
            out = await ChunkLookup.afetch_rows_in_ranges(
                MagicMock(table_name="data_1_x"),
                [("f1", 3, 5), ("f1", 5, 7), ("f1", 20, 22), ("f2", -1, 1)])
        sql, params = conn.calls[0]
        assert sql.count("BETWEEN") == 3                     # f1 3..7, f1 20..22, f2 0..1
        assert (params["s_0"], params["e_0"]) == (3, 7) and params["s_2"] == 0
        assert "parent_node_id" in sql and '"data_1_x"' in sql
        assert out == {"f1": [(3, "t3", "P"), (4, "t4", "P")], "f2": [(0, "u0", None)]}

    async def test_empty_request_and_db_failure(self):
        from src.domain.rag.chunk_lookup import ChunkLookup

        assert await ChunkLookup.afetch_rows_in_ranges(MagicMock(), []) == {}
        engine = MagicMock()
        engine.connect.return_value = _AsyncConn(fail=True)
        with patch("db.db.get_async_engine", return_value=engine):
            assert await ChunkLookup.afetch_rows_in_ranges(
                MagicMock(table_name="data_1_x"), [("f1", 0, 1)]) == {}
