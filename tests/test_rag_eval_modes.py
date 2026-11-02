"""The eval harness scores through the production search path (no private copy).

Its BM25 mode used to send the raw question to llama-index's sparse mode (no
Chinese segmentation) and its "hybrid" mode was llama-index's native hybrid,
identical to vector. Every mode now calls HybridSearcher.
"""
import importlib.util
import os
from unittest.mock import AsyncMock, MagicMock

from src.domain.rag.hybrid_search import HybridSearcher, SearchOutcome, SearchSettings
from tests.test_hybrid_search import nws

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "rag_eval.py")


def _load():
    spec = importlib.util.spec_from_file_location("rag_eval_modes", _PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _searcher():
    s = HybridSearcher(MagicMock(), index=MagicMock())
    s.dense = AsyncMock(return_value=[nws("d1"), nws("d2")])
    s.sparse = AsyncMock(return_value=[nws("b1"), nws("d2")])
    s.search = AsyncMock(return_value=SearchOutcome(nodes=[nws("r1")]))
    return s


async def test_modes_route_to_the_shared_searcher():
    m = _load()
    assert m.BASE_MODES == ("vector", "bm25", "rrf")
    s = _searcher()
    st = SearchSettings(top_k=10, sparse_top_k=30, fusion="concat")
    assert [n.node.node_id for n in await m._retrieve_nodes(s, st, "q", "vector", 1)] == ["d1"]
    await m._retrieve_nodes(s, st, "q", "bm25", 10)
    assert s.sparse.await_args.args[1] == 30            # honours sparse_top_k
    rrf = await m._retrieve_nodes(s, st, "q", "rrf", 10)
    assert rrf[0].node.node_id == "d2"                   # fused with RRF even if config says concat
    assert [n.node.node_id for n in await m._retrieve_nodes(s, st, "q", "rerank", 10)] == ["r1"]
    assert s.search.await_args.args[1].top_k == 10
    assert await m._retrieve_nodes(s, st, "q", "hybrid", 10) == []


async def test_rank_of_gold():
    m = _load()
    s = _searcher()
    st = SearchSettings()
    assert await m._rank_of_gold(s, st, "q", "d2", "vector", 5) == 2
    assert await m._rank_of_gold(s, st, "q", "zz", "vector", 5) is None
