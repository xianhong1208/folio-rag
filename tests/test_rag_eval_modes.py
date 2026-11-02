"""Eval modes: every mode scores through the production search path (no private copy).

vector / bm25 / rrf / rerank call the shared HybridSearcher (the same one REST, MCP
and the console use); production is the very adapter call POST /api/rag/query makes.
The old "hybrid" mode (llama-index's native concat, identical to vector) is gone, the
BM25 mode no longer sends the raw question to llama-index's unsegmented sparse mode,
and deltas only compare comparable runs.
"""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.rag.hybrid_search import SearchOutcome, SearchSettings
from tests.test_hybrid_search import nws

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rag_eval.py"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("rag_eval_modes_mod", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_modes_are_labelled_and_hybrid_is_gone(ev):
    assert ev.BASE_MODES == ("vector", "bm25", "rrf")
    assert ev.MODES == ("vector", "bm25", "rrf", "rerank", "production")
    assert "hybrid" not in ev.MODES and set(ev.MODE_LABELS) == set(ev.MODES)
    assert ev.BEST_MODE == "production"


class TestRetrievers:
    def _build(self, ev, reranker="RR"):
        adapter = MagicMock()
        adapter._query_svc._ctx.get_vector_store.return_value = "VS"
        adapter._query_svc._ctx.reranker = reranker
        searcher = MagicMock()
        searcher.dense = AsyncMock(return_value=[nws("d1"), nws("d2")])
        searcher.sparse = AsyncMock(return_value=[nws("b1"), nws("d2")])
        searcher.candidates = AsyncMock(return_value=([nws("f1"), nws("f2")], [], []))
        searcher.search = AsyncMock(return_value=SearchOutcome(nodes=[nws("r1")]))
        defaults = {"top_k": 5, "similarity_cutoff": 0.3, "sparse_top_k": 20, "hybrid_alpha": 0.5}
        settings = SearchSettings(top_k=10, sparse_top_k=30, fusion="concat")
        with patch("src.adapter.rag.get_rag_adapter", return_value=adapter), \
             patch("src.api.router.rag_query._resolve_retrieval_defaults", return_value=defaults), \
             patch("src.domain.rag.hybrid_search.SearchSettings.from_config",
                   return_value=settings), \
             patch("src.domain.rag.hybrid_search.HybridSearcher",
                   return_value=searcher) as hs:
            r = ev._Retrievers(SimpleNamespace(id=7, user_token="tok", name="f"), k=3)
        return r, adapter, searcher, hs

    def test_searcher_is_built_on_the_folder_store_with_the_reranker(self, ev):
        _, _, _, hs = self._build(ev)
        assert hs.call_args.args == ("VS",) and hs.call_args.kwargs == {"reranker": "RR"}

    def test_production_is_the_rest_query_call(self, ev):
        r, adapter, _, _ = self._build(ev)
        res = SimpleNamespace(results=[
            SimpleNamespace(metadata=SimpleNamespace(node_id=f"n{i}", node_role="leaf"),
                            text=f"t{i}") for i in range(5)])
        adapter.query_rag = AsyncMock(return_value=res)
        got = asyncio.run(r.nodes("q", "production"))
        assert got == [("n0", "t0", None), ("n1", "t1", None), ("n2", "t2", None)]
        kw = adapter.query_rag.await_args.kwargs
        assert kw == {"query": "q", "folder_id": 7, "token": "tok", "top_k": 3,
                      "similarity_cutoff": 0.3, "use_cache": False, "sparse_top_k": 20,
                      "hybrid_alpha": 0.5}

    def test_production_marks_merged_sections(self, ev):
        r, adapter, _, _ = self._build(ev)
        res = SimpleNamespace(results=[
            SimpleNamespace(metadata=SimpleNamespace(node_id="P", node_role="parent"), text="s")])
        adapter.query_rag = AsyncMock(return_value=res)
        assert asyncio.run(r.nodes("q", "production")) == [("P", "s", "P")]

    def test_modes_route_to_the_shared_searcher(self, ev):
        r, _, s, _ = self._build(ev)
        assert [n for n, _, _ in asyncio.run(r.nodes("q", "vector"))] == ["d1", "d2"]
        assert s.dense.await_args.args == ("q", 3)
        asyncio.run(r.nodes("q", "bm25"))
        assert s.sparse.await_args.args[1] == 30            # honours sparse_top_k
        assert [n for n, _, _ in asyncio.run(r.nodes("q", "rrf"))] == ["f1", "f2"]
        assert s.candidates.await_args.args[1].fusion == "rrf"   # even if config says concat
        assert [n for n, _, _ in asyncio.run(r.nodes("q", "rerank"))] == ["r1"]
        assert s.search.await_args.args[1].top_k == 3

    def test_rerank_mode_only_with_a_reranker(self, ev):
        r, _, _, _ = self._build(ev)
        assert r.modes() == list(ev.MODES)
        r, _, _, _ = self._build(ev, reranker=None)
        assert r.modes() == ["vector", "bm25", "rrf", "production"]

    def test_unknown_mode_rejected(self, ev):
        r, _, _, _ = self._build(ev)
        with pytest.raises(ValueError):
            asyncio.run(r.nodes("q", "hybrid"))


def test_rank_in_matches_gold_leaf_or_its_merged_section(ev):
    hits = [("x", "", None), ("P", "", "P"), ("g", "", None)]
    assert ev._rank_in(hits, "g") == 3
    assert ev._rank_in(hits, "leaf-under-P", "P") == 2
    assert ev._rank_in(hits, "zz", None) is None


def test_evaluate_scores_each_mode(ev):
    class FakeRetr:
        def __init__(self, folder, k):
            pass

        def modes(self):
            return ["vector", "bm25", "rrf", "production"]

        async def nodes(self, q, mode):
            return {"vector": [("x", "", None), ("g1", "", None)], "bm25": [("g1", "", None)],
                    "rrf": [], "production": [("S", "ctx", "S")]}[mode]

    items = [{"query": "q1", "gold_node_id": "g1"}, {"query": "q2", "gold_node_id": "g2"}]
    with patch.object(ev, "_Retrievers", FakeRetr), \
         patch.object(ev, "_gold_parents", return_value={"g1": "S"}), \
         patch.object(ev, "_config_snapshot", return_value={}):
        rep = asyncio.run(ev.evaluate(SimpleNamespace(name="f"), items, k=5))
    assert rep["modes"] == ["vector", "bm25", "rrf", "production"]
    assert rep["summary"]["production"]["recall@5"] == 0.5   # g1 inside merged section S
    assert rep["summary"]["vector"]["mrr"] == 0.25  # rank 2 for one of two questions
    assert rep["summary"]["rrf"]["recall@5"] == 0.0
    assert rep["harness"] == ev.HARNESS_VERSION
    assert rep["question_set"] == ev.question_set_id(items)
    assert rep["mode_labels"]["production"].startswith("Production")


def test_gold_parents_reads_parent_ids(ev):
    with patch.object(ev, "_query_vector_table", return_value=[("g1", "P"), ("g2", None)]) as q:
        assert ev._gold_parents(SimpleNamespace(), [{"gold_node_id": "g1"},
                                                    {"gold_node_id": "g2"}]) == {"g1": "P"}
    assert "parent_node_id" in q.call_args.args[1]
    assert ev._gold_parents(SimpleNamespace(), []) == {}
    with patch.object(ev, "_query_vector_table", side_effect=RuntimeError("db down")):
        assert ev._gold_parents(SimpleNamespace(), [{"gold_node_id": "g1"}]) == {}


class TestComparablePrevious:
    def test_question_set_id_is_order_independent(self, ev):
        a = [{"query": "x", "gold_node_id": "1"}, {"query": "y", "gold_node_id": "2"}]
        assert ev.question_set_id(a) == ev.question_set_id(list(reversed(a)))
        assert ev.question_set_id(a) != ev.question_set_id(a[:1])

    def test_only_same_set_k_and_harness(self, ev):
        rep = {"harness": 3, "question_set": "abc", "k": 10}
        hist = [
            {"harness": 3, "question_set": "abc", "k": 10, "n": 1},
            {"harness": 3, "question_set": "abc", "k": 5, "n": 2},     # other k
            {"harness": 3, "question_set": "zzz", "k": 10, "n": 3},    # other questions
            {"harness": 2, "question_set": "abc", "k": 10, "n": 4},    # older mode set
            {"question_set": "abc", "k": 10, "n": 5},                  # old harness (hybrid era)
        ]
        assert ev.comparable_previous(hist, rep)["n"] == 1
        assert ev.comparable_previous(hist[1:], rep) is None
        assert ev.comparable_previous([], rep) is None

    def test_history_row_records_comparability_keys(self, ev, tmp_path, monkeypatch):
        monkeypatch.setattr(ev, "_history_path", lambda slug: str(tmp_path / "h.jsonl"))
        ev._append_history("s", {"k": 10, "n": 3, "summary": {}, "modes": ["vector"],
                                 "harness": 3, "question_set": "abc"})
        row = ev._read_history("s")[0]
        assert row["harness"] == 3 and row["question_set"] == "abc" and row["modes"] == ["vector"]
