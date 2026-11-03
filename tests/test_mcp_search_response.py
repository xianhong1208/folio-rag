"""MCP search response: size budget, score_type, hints and top_k bounds.

Includes an offline measurement: a response built from synthetic hits that
mimic the live problem (each hit expanded to ±5 neighbours, adjacent hits, long
CJK text) must stay within the token budget and must not repeat text.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.rag import agentic_handlers as ah
from src.domain.rag.auto_merging import AutoMergingRetriever, MergedResult
from src.domain.rag.hybrid_search import SearchOutcome
from src.domain.rag.token_budget import estimate_tokens
from tests.test_hybrid_search import nws

FOLDER = SimpleNamespace(id=1, name="docs", vector_table_uuid="u")


def _mr(i, text, score=None, role="expanded", merged=None):
    return MergedResult(text=text, score=score if score is not None else 0.9 - i / 100,
                        file_id="f1", file_name="a.pdf", node_id=f"n{i}", node_role=role,
                        chunk_index_range=[i, i], merged_from_leaves=merged or [])


def _retriever(results, outcome=None):
    r = MagicMock()
    r.aquery = AsyncMock(return_value=results)
    r.last_outcome = outcome
    return r


async def _search(results, top_k=5, budget=None, outcome=None, **kw):
    return await ah.handle_search(query="年假規定", folder=FOLDER,
                                  retriever=_retriever(results, outcome), top_k=top_k,
                                  expand_context=True, token_budget=budget, **kw)


class TestBudget:
    async def test_lowest_ranked_results_are_dropped_and_reported(self):
        results = [_mr(i, "年假" * 600) for i in range(6)]       # ~840 tokens each
        out = await _search(results, top_k=6, budget=2000)
        assert [r["node_id"] for r in out["results"]] == ["n0", "n1"]
        assert out["omitted_results"] == 4 and out["total_results"] == 2
        assert out["response_tokens_estimated"] <= 2000
        assert "4 lower-ranked result(s) were left out" in out["_hint"]

    async def test_oversized_best_result_is_cut_not_dropped(self):
        out = await _search([_mr(0, "長" * 20000)], top_k=1, budget=1000)
        (r,) = out["results"]
        assert r["text_truncated"] is True and 0 < len(r["text"]) < 20000
        assert out["response_tokens_estimated"] <= 1000

    async def test_zero_budget_disables_the_cap(self):
        results = [_mr(i, "x" * 8000) for i in range(3)]
        out = await _search(results, top_k=3, budget=0)
        assert len(out["results"]) == 3 and "omitted_results" not in out

    async def test_budget_defaults_to_config(self):
        cfg = SimpleNamespace(rag=SimpleNamespace(retrieval=SimpleNamespace(
            mcp_response_token_budget=1500)))
        with patch("src.config.config_manager.Config.get_config_model", return_value=cfg):
            out = await _search([_mr(i, "字" * 1000) for i in range(5)])
        assert out["response_tokens_estimated"] <= 1500 and out["omitted_results"] > 0

    def test_configured_budget_fallbacks(self):
        with patch("src.config.config_manager.Config.get_config_model",
                   side_effect=RuntimeError):
            assert ah.configured_response_budget() == ah.DEFAULT_RESPONSE_BUDGET
        cfg = SimpleNamespace(rag=SimpleNamespace(retrieval=SimpleNamespace(
            mcp_response_token_budget=None)))
        with patch("src.config.config_manager.Config.get_config_model", return_value=cfg):
            assert ah.configured_response_budget() == ah.DEFAULT_RESPONSE_BUDGET

    def test_fit_to_budget_with_no_results(self):
        resp = ah.fit_to_budget({"mode": "search", "results": []}, 100)
        assert resp["response_tokens_estimated"] > 0


class TestContract:
    async def test_score_type_reported(self):
        out = await _search([_mr(0, "a")], outcome=SearchOutcome(nodes=[], reranked=True))
        assert out["score_type"] == "rerank"
        out = await _search([_mr(0, "a")], outcome=SearchOutcome(nodes=[]))
        assert out["score_type"] == "rrf"
        out = await _search([_mr(0, "a")], outcome=None)
        assert out["score_type"] == "rrf"

    async def test_top_k_is_bounded(self):
        r = _retriever([])
        await ah.handle_search(query="hello", folder=FOLDER, retriever=r, top_k=500,
                               expand_context=False)
        assert r.aquery.await_args.kwargs["top_k"] == 20
        await ah.handle_search(query="hello", folder=FOLDER, retriever=r, top_k=0,
                               expand_context=False)
        assert r.aquery.await_args.kwargs["top_k"] == 1
        assert ah.clamp_top_k("x") == 1 and ah.clamp_top_k(None) == 1

    async def test_similarity_cutoff_is_ignored(self):
        r = _retriever([])
        await ah.handle_search(query="hello", folder=FOLDER, retriever=r, top_k=3,
                               similarity_cutoff=0.9, expand_context=False)
        assert "similarity_cutoff" not in r.aquery.await_args.kwargs

    @pytest.mark.parametrize("n, needle", [(0, "No matching passages"),
                                           (2, "Only 2 passage(s) matched"),
                                           (5, "ranked best first")])
    async def test_hints_never_push_read_for_weak_results(self, n, needle):
        out = await _search([_mr(i, "a") for i in range(n)], top_k=5)
        hint = out["_hint"]
        assert needle in hint
        assert "similarity_cutoff" not in hint and "lower" not in hint.lower()
        if n < 5:                                   # weak results: never steer to read
            assert "mode='read'" not in hint

    async def test_merged_window_reports_its_hits(self):
        out = await _search([_mr(0, "a", merged=["x", "y"]),
                             _mr(1, "b", role="parent", merged=["p", "q", "r"])])
        assert out["results"][0]["merged_hits"] == 2
        assert out["results"][1]["merged_from_leaves"] == 3

    async def test_invalid_queries(self):
        r = _retriever([])
        too_short = await ah.handle_search(query="a", folder=FOLDER, retriever=r, top_k=1,
                                           expand_context=False)
        assert "too short" in too_short["error"]
        too_long = await ah.handle_search(query="x" * 5000, folder=FOLDER, retriever=r,
                                          top_k=1, expand_context=False)
        assert "too long" in too_long["error"]

    async def test_retriever_failure_propagates(self):
        r = MagicMock()
        r.aquery = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            await ah.handle_search(query="hello", folder=FOLDER, retriever=r, top_k=1,
                                   expand_context=False)


# --------------------------------------------------- offline measurement

def _synthetic_chunk(i: int) -> str:
    # ~170 CJK characters + a little Latin, like a Docling leaf of a policy PDF.
    return (f"第{i}條 員工年假規定：到職滿一年者享有特休假七日，" * 5
            + f"(ref policy-{i:03d})")


class TestOfflineMeasurement:
    """End to end through AutoMergingRetriever + handle_search with synthetic hits."""

    async def _response(self, n_neighbours: int, budget: int):
        # 10 hits, in adjacent pairs (4,5), (9,10) … inside one 60-leaf file whose
        # parents are runs of 6 leaves.
        hit_idx = [4, 5, 9, 10, 16, 17, 23, 30, 31, 44]
        leaves = [nws(f"leaf{i}", 0.99 - k / 100, chunk_index=i, parent=f"P{i // 6}",
                      text=_synthetic_chunk(i), file_id="f1", file_name="policy.pdf")
                  for k, i in enumerate(hit_idx)]
        table = {"f1": [(i, _synthetic_chunk(i), f"P{i // 6}") for i in range(60)]}

        async def fetch(vs, ranges):
            out = {}
            for fid, lo, hi in ranges:
                out.setdefault(fid, [])
                out[fid] += [r for r in table[fid] if lo <= r[0] <= hi and r not in out[fid]]
            return out

        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=SearchOutcome(nodes=leaves, reranked=True))
        retr = AutoMergingRetriever(MagicMock(), merge_threshold=1.1,
                                    expand_context_neighbors=n_neighbours, searcher=searcher)
        with patch("src.domain.rag.chunk_lookup.ChunkLookup.afetch_rows_in_ranges",
                   new=AsyncMock(side_effect=fetch)):
            resp = await ah.handle_search(query="年假規定", folder=FOLDER, retriever=retr,
                                          top_k=10, expand_context=True, token_budget=budget)
        return resp, json.dumps(resp, ensure_ascii=False)

    async def test_no_chunk_is_returned_twice(self):
        resp, _ = await self._response(n_neighbours=5, budget=0)
        chunks = [c for r in resp["results"] for c in r["text"].split("\n\n")]
        assert len(chunks) == len(set(chunks))
        # adjacent hits share one result
        assert len(resp["results"]) < 10

    async def test_windows_stay_in_the_hit_section(self):
        resp, _ = await self._response(n_neighbours=5, budget=0)
        for r in resp["results"]:
            lo, hi = r["chunk_index_range"]
            assert lo // 6 == hi // 6, r["chunk_index_range"]       # one section each

    async def test_default_budget_holds_and_is_far_below_the_old_size(self):
        resp, raw = await self._response(n_neighbours=1, budget=6000)
        assert estimate_tokens(raw) <= 6000
        assert resp["response_tokens_estimated"] <= 6000
        # The pre-fix shape: every hit expanded ±5 with no merging and no budget.
        old_tokens = sum(estimate_tokens("\n\n".join(_synthetic_chunk(j) for j in
                                                      range(max(i - 5, 0), i + 6)))
                         for i in [4, 5, 9, 10, 16, 17, 23, 30, 31, 44])
        assert old_tokens > 3 * estimate_tokens(raw)
