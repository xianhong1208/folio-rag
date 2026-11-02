"""The REST query cache must key on everything that changes results, and be
cleared when a runtime setting that changes results is edited."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.domain.rag.hybrid_search import SearchSettings, cache_fingerprint
from src.infrastructure.cache.cache_service import CacheKeys, CacheService


def _rr(**kw):
    base = dict(_model="m", _score_threshold=0.5, _query_template="{text}",
                _document_template="{text}")
    base.update(kw)
    return SimpleNamespace(**base)


def _fp(settings=None, reranker=None, merge=0.5, segmenter="ckip", query="q"):
    with patch("src.domain.rag.text_segmenter.configured_segmenter_name", return_value=segmenter):
        return cache_fingerprint(query, settings or SearchSettings(), reranker=reranker,
                                 merge_threshold=merge)


class TestFingerprint:
    def test_stable(self):
        assert _fp() == _fp()

    @pytest.mark.parametrize("change", [
        dict(settings=SearchSettings(fusion="concat")),
        dict(settings=SearchSettings(rrf_k=10)),
        dict(settings=SearchSettings(top_k=9)),
        dict(settings=SearchSettings(sparse_top_k=40)),
        dict(settings=SearchSettings(similarity_cutoff=0.3)),
        dict(settings=SearchSettings(text_search_config="english")),
        dict(reranker=_rr()),
        dict(merge=1.1),
        dict(segmenter="bigram"),
        dict(query="other"),
    ])
    def test_every_result_affecting_input_changes_the_key(self, change):
        assert _fp(**change) != _fp()

    @pytest.mark.parametrize("kw", [
        dict(_score_threshold=0.25), dict(_model="other"),
        dict(_query_template="<|im_start|>{text}"), dict(_document_template="d: {text}"),
    ])
    def test_reranker_settings_change_the_key(self, kw):
        assert _fp(reranker=_rr(**kw)) != _fp(reranker=_rr())

    def test_segmenter_ignored_outside_simple_config(self):
        st = SearchSettings(text_search_config="english")
        assert _fp(settings=st, segmenter="bigram") == _fp(settings=st, segmenter="ckip")


class TestInvalidation:
    def setup_method(self):
        CacheService.reset_instance()

    def test_clear_query_cache_drops_only_query_entries(self):
        from src.adapter.rag_context import RAGContext
        cache = CacheService.get_instance()
        cache.set(CacheKeys.rag_query(1, "abc"), "r1")
        cache.set(CacheKeys.rag_query(2, "def"), "r2")
        cache.set(CacheKeys.folder_by_id(1), "folder")
        RAGContext.clear_query_cache(MagicMock())
        assert cache.get(CacheKeys.rag_query(1, "abc")) is None
        assert cache.get(CacheKeys.rag_query(2, "def")) is None
        assert cache.get(CacheKeys.folder_by_id(1)) == "folder"

    @pytest.mark.parametrize("groups, cleared", [
        (["rag.retrieval"], True), (["rag.rerank"], True), (["rag.embedding"], True),
        (["rag.asr"], False), (["rag.llm"], False),
    ])
    def test_runtime_changes_clear_the_query_cache(self, groups, cleared):
        from src.adapter.rag_context import RAGContext
        ctx = MagicMock()
        cfg = SimpleNamespace(rag=SimpleNamespace())
        with patch("src.adapter.rag_context.Config.get_config_model", return_value=cfg):
            RAGContext.apply_runtime_changes(ctx, groups)
        assert ctx.clear_query_cache.called is cleared
