"""Rerank policy: mxbai chat template, threshold + top score, failure handling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock


from src.config.model import RerankConfig
from src.domain.rag import rerank_policy as rp
from src.domain.rag.hybrid_search import SearchSettings
from tests.test_hybrid_search import nws, searcher_with

MXBAI = "mixedbread-ai/mxbai-rerank-base-v2"


class TestReranker:
    """ref: a failed rerank must not hand back all 3 × top_k candidates."""

    def _reranker(self):
        from src.domain.rag.reranker import Reranker
        return Reranker(base_url="http://x", model="m", score_threshold=0.3)

    async def test_arerank_failure_truncates_to_top_n(self):
        r = self._reranker()
        r._aget_scores = AsyncMock(side_effect=RuntimeError("down"))
        nodes = [nws(c) for c in "abcdef"]
        assert [n.node.node_id for n in await r.arerank("q", nodes, top_n=2)] == ["a", "b"]

    def test_rerank_failure_truncates_to_top_n(self):
        r = self._reranker()
        r._get_scores = MagicMock(side_effect=RuntimeError("down"))
        nodes = [nws(c) for c in "abcdef"]
        assert len(r.rerank("q", nodes, top_n=3)) == 3

    async def test_ascore_and_threshold_are_public(self):
        r = self._reranker()
        r._aget_scores = AsyncMock(return_value=[0.1, 0.9])
        assert await r.ascore("q", ["x", "y"]) == [0.1, 0.9]
        assert r.score_threshold == 0.3


class TestTemplates:
    def test_mxbai_default_or_legacy_templates_get_the_chat_format(self):
        for qt, dt in (("{text}", "{text}"),
                       ("query: {text}", "document: {text}\nYou are a search relevance expert…")):
            q, d = rp.effective_templates(MXBAI, qt, dt)
            assert q == rp.MXBAI_V2_QUERY_TEMPLATE and d == rp.MXBAI_V2_DOCUMENT_TEMPLATE

    def test_joined_prompt_matches_the_official_format(self):
        q, d = rp.effective_templates(MXBAI, "{text}", "{text}")
        prompt = q.replace("{text}", "What is {x}?") + d.replace("{text}", "Doc {y}")
        assert prompt == (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
            "assistant.<|im_end|>\n<|im_start|>user\nquery: What is {x}?\ndocument: Doc {y}\n"
            "You are a search relevance expert who evaluates how well documents match search "
            "queries. For each query-document pair, carefully analyze the semantic relationship "
            "between them, then provide your binary relevance judgment (0 for not relevant, "
            "1 for relevant).\nRelevance:<|im_end|>\n<|im_start|>assistant\n")

    def test_left_alone_when_off_custom_or_other_models(self):
        assert rp.effective_templates(MXBAI, "{text}", "{text}", auto=False) == ("{text}", "{text}")
        assert rp.effective_templates(MXBAI, "Q: {text}", "D: {text}") == ("Q: {text}", "D: {text}")
        assert rp.effective_templates("BAAI/bge-reranker-v2-m3", "{text}", "{text}") == ("{text}", "{text}")
        chat = ("<|im_start|>user {text}", "{text}")
        assert rp.effective_templates(MXBAI, *chat) == chat

    def test_resolve_config(self):
        cfg = RerankConfig(enabled=True, model=MXBAI)
        out = rp.resolve_rerank_config(cfg)
        assert out.query_template == rp.MXBAI_V2_QUERY_TEMPLATE and cfg.query_template == "{text}"
        bge = RerankConfig(enabled=True)
        assert rp.resolve_rerank_config(bge) is bge
        assert rp.resolve_rerank_config(None) is None
        plain = SimpleNamespace(model=MXBAI, query_template="{text}", document_template="{text}",
                                auto_template=True)
        assert rp.resolve_rerank_config(plain).document_template == rp.MXBAI_V2_DOCUMENT_TEMPLATE

    def test_auto_template_is_a_config_switch(self):
        assert RerankConfig().auto_template is True
        from src.config.runtime_overrides import EDITABLE
        assert EDITABLE["rag.rerank.auto_template"] == "live"


class _Scorer:
    def __init__(self, table=None, threshold=0.0, fail=False, wrong_len=False):
        self.table, self.score_threshold, self.fail, self.wrong_len = table or {}, threshold, fail, wrong_len

    async def ascore(self, query, texts):
        if self.fail:
            raise RuntimeError("down")
        out = [self.table.get(t.split()[-1], 0.0) for t in texts]
        return out[:-1] if self.wrong_len else out


class TestRerank:
    async def test_threshold_filters_but_top_score_is_kept(self):
        r = _Scorer({"a": 0.2, "b": 0.7, "c": 0.4}, threshold=0.5)
        out = await rp.rerank(r, "q", [nws("a"), nws("b"), nws("c")], top_n=2)
        assert [n.node.node_id for n in out.nodes] == ["b"] and out.top_score == 0.7
        out = await rp.rerank(_Scorer({"a": 0.2}, threshold=0.5), "q", [nws("a")], top_n=2)
        assert out.nodes == [] and out.top_score == 0.2 and not out.failed

    async def test_failure_keeps_top_n_in_order_and_is_flagged(self):
        for r in (_Scorer(fail=True), _Scorer(wrong_len=True)):
            out = await rp.rerank(r, "q", [nws(c) for c in "abcdef"], top_n=2)
            assert [n.node.node_id for n in out.nodes] == ["a", "b"]
            assert out.failed and out.top_score is None

    async def test_scores_the_retrieval_text_not_the_display_text(self):
        seen = {}

        class _Spy:
            score_threshold = 0.0

            async def ascore(self, query, texts):
                seen["texts"] = texts
                return [0.5] * len(texts)

        node = nws("a", text="36-month warranty.", breadcrumb="Aurora › Warranty",
                   context_prefix="Warranty terms of the Aurora router.")
        out = await rp.rerank(_Spy(), "q", [node], top_n=1)
        assert seen["texts"] == ["Warranty terms of the Aurora router.\nAurora › Warranty\n"
                                 "36-month warranty."]
        assert out.nodes[0].node.text == "36-month warranty."  # display text untouched

    async def test_rerankers_without_ascore_use_arerank(self):
        r = MagicMock(spec=["arerank"])
        r.arerank = AsyncMock(return_value=[nws("b", 0.9), nws("a", 0.1)])
        out = await rp.rerank(r, "q", [nws("a"), nws("b")], top_n=1)
        assert [n.node.node_id for n in out.nodes] == ["b"] and out.top_score == 0.9

    async def test_empty(self):
        assert (await rp.rerank(_Scorer(), "q", [], 3)).nodes == []


class TestSearchOutcome:
    async def test_search_reports_top_score_and_failure(self):
        s = searcher_with([nws("a"), nws("b")], [], reranker=_Scorer({"a": 0.3, "b": 0.6}, 0.5))
        out = await s.search("q", SearchSettings(top_k=2))
        assert out.reranked and out.top_score == 0.6 and [n.node.node_id for n in out.nodes] == ["b"]
        s = searcher_with([nws("a"), nws("b")], [], reranker=_Scorer(fail=True))
        out = await s.search("q", SearchSettings(top_k=1))
        assert out.rerank_failed and not out.reranked and len(out.nodes) == 1
