"""Eval modes: vector / bm25 / production (no "hybrid"), production goes through the
same adapter call as POST /api/rag/query, and deltas only compare comparable runs."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rag_eval.py"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("rag_eval_modes_mod", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _node(nid, text="t"):
    return SimpleNamespace(node=SimpleNamespace(node_id=nid, text=text))


def test_modes_are_labelled_and_hybrid_is_gone(ev):
    assert ev.MODES == ("vector", "bm25", "production")
    assert "hybrid" not in ev.MODES and set(ev.MODE_LABELS) == set(ev.MODES)
    assert ev.BEST_MODE == "production"


class TestRetrievers:
    def _build(self, ev):
        adapter = MagicMock()
        adapter._query_svc._ctx.get_vector_store.return_value = "VS"
        engine = MagicMock()
        defaults = {"top_k": 5, "similarity_cutoff": 0.3, "sparse_top_k": 20, "hybrid_alpha": 0.5}
        with patch("src.adapter.rag.get_rag_adapter", return_value=adapter), \
             patch("src.api.router.rag_query._resolve_retrieval_defaults", return_value=defaults), \
             patch("src.adapter.rag_query._fusion_config", return_value=("rrf", 60, "simple")), \
             patch("src.domain.rag.query_engine.QueryEngine", return_value=engine) as qe:
            r = ev._Retrievers(SimpleNamespace(id=7, user_token="tok", name="f"), k=3)
        return r, adapter, engine, qe

    def test_production_is_the_rest_query_call(self, ev):
        r, adapter, _, _ = self._build(ev)
        res = SimpleNamespace(results=[SimpleNamespace(metadata=SimpleNamespace(node_id=f"n{i}"),
                                                       text=f"t{i}") for i in range(5)])
        adapter.query_rag = AsyncMock(return_value=res)
        got = asyncio.run(r.nodes("q", "production"))
        assert got == [("n0", "t0"), ("n1", "t1"), ("n2", "t2")]
        kw = adapter.query_rag.await_args.kwargs
        assert kw == {"query": "q", "folder_id": 7, "token": "tok", "top_k": 3,
                      "similarity_cutoff": 0.3, "use_cache": False, "sparse_top_k": 20,
                      "hybrid_alpha": 0.5}

    def test_single_signal_modes_use_the_production_engine_paths(self, ev):
        r, _, engine, qe = self._build(ev)
        assert qe.call_args.kwargs["reranker"] is None
        assert qe.call_args.kwargs["text_search_config"] == "simple"
        engine._aretrieve_mode = AsyncMock(return_value=[_node("a"), _node("b")])
        assert [n for n, _ in asyncio.run(r.nodes("q", "vector"))] == ["a", "b"]
        assert engine._aretrieve_mode.await_args.args == ("q", "default", 3)
        asyncio.run(r.nodes("q", "bm25"))
        # "sparse" goes through _aretrieve_mode, i.e. the CKIP OR-query production uses.
        assert engine._aretrieve_mode.await_args.args == ("q", "sparse", 3)

    def test_unknown_mode_rejected(self, ev):
        r, _, _, _ = self._build(ev)
        with pytest.raises(ValueError):
            asyncio.run(r.nodes("q", "hybrid"))


def test_evaluate_scores_each_mode(ev):
    class FakeRetr:
        def __init__(self, folder, k):
            pass

        async def nodes(self, q, mode):
            return {"vector": [("x", ""), ("g1", "")], "bm25": [("g1", "")],
                    "production": [("g1", "ctx")]}[mode]

    items = [{"query": "q1", "gold_node_id": "g1"}, {"query": "q2", "gold_node_id": "g2"}]
    with patch.object(ev, "_Retrievers", FakeRetr), patch.object(ev, "_config_snapshot", return_value={}):
        rep = asyncio.run(ev.evaluate(SimpleNamespace(name="f"), items, k=5))
    assert rep["modes"] == ["vector", "bm25", "production"]
    assert rep["summary"]["production"]["recall@5"] == 0.5
    assert rep["summary"]["vector"]["mrr"] == 0.25  # rank 2 for one of two questions
    assert rep["harness"] == ev.HARNESS_VERSION
    assert rep["question_set"] == ev.question_set_id(items)
    assert rep["mode_labels"]["production"].startswith("Production")


class TestComparablePrevious:
    def test_question_set_id_is_order_independent(self, ev):
        a = [{"query": "x", "gold_node_id": "1"}, {"query": "y", "gold_node_id": "2"}]
        assert ev.question_set_id(a) == ev.question_set_id(list(reversed(a)))
        assert ev.question_set_id(a) != ev.question_set_id(a[:1])

    def test_only_same_set_k_and_harness(self, ev):
        rep = {"harness": 2, "question_set": "abc", "k": 10}
        hist = [
            {"harness": 2, "question_set": "abc", "k": 10, "n": 1},
            {"harness": 2, "question_set": "abc", "k": 5, "n": 2},     # other k
            {"harness": 2, "question_set": "zzz", "k": 10, "n": 3},    # other questions
            {"question_set": "abc", "k": 10, "n": 4},                  # old harness (hybrid era)
        ]
        assert ev.comparable_previous(hist, rep)["n"] == 1
        assert ev.comparable_previous(hist[1:], rep) is None
        assert ev.comparable_previous([], rep) is None

    def test_history_row_records_comparability_keys(self, ev, tmp_path, monkeypatch):
        monkeypatch.setattr(ev, "_history_path", lambda slug: str(tmp_path / "h.jsonl"))
        ev._append_history("s", {"k": 10, "n": 3, "summary": {}, "modes": ["vector"],
                                 "harness": 2, "question_set": "abc"})
        row = ev._read_history("s")[0]
        assert row["harness"] == 2 and row["question_set"] == "abc" and row["modes"] == ["vector"]
