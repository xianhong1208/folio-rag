"""Rerank and Rebuild-FTS use the retrieval text (prefix + breadcrumb + original text)."""

from types import SimpleNamespace

from src.domain.rag import vector_store_manager as vsm
from src.domain.rag.reranker import Reranker


def _nws(text, meta):
    return SimpleNamespace(node=SimpleNamespace(text=text, metadata=meta), score=0.0, text=text)


def test_rerank_scores_retrieval_text():
    rr = Reranker.__new__(Reranker)
    rr._top_n, rr._score_threshold = 5, 0.0
    seen = {}

    def _scores(query, docs):
        seen["docs"] = docs
        return [0.1, 0.9]

    rr._get_scores = _scores
    nodes = [_nws("36-month warranty.", {"breadcrumb": "Aurora › Warranty"}),
             _nws("old\n\nrow", {})]
    out = rr.rerank("q", nodes)
    assert seen["docs"] == ["Aurora › Warranty\n36-month warranty.", "old\n\nrow"]
    assert out[0].node.text == "old\n\nrow"  # display text untouched


def test_rebuild_fts_selects_retrieval_text():
    executed = []

    class _Conn:
        def execute(self, stmt, params=None):
            executed.append(str(stmt))
            return SimpleNamespace(fetchall=lambda: [])

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    engine = SimpleNamespace(connect=lambda: _Conn())
    assert vsm.rebuild_text_search_tsv(engine, "data_1_x", segmenter=None, file_id="f") == 0
    select = next(s for s in executed if s.startswith("SELECT"))
    assert "concat_ws" in select and "breadcrumb" in select
