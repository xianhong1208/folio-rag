
"""Behavior contract for three-mode query orchestration (query_agentic).

Pins down two invariants: the ACL runs before any data (vector store) access,
and list/search/read each dispatch correctly while an unknown mode returns an
error dict.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.adapter.rag_query import RAGQueryService
from src.domain.exceptions import UnauthorizedAccessError

# Orchestration and ACL live in domain (folder_acl / agentic_handlers).
_ACL = "src.domain.rag.folder_acl"
_TH = "src.domain.rag.agentic_handlers"
_RQ = "src.adapter.rag_query"


def _make_service():
    """Fake ctx: vector_store_manager.get_or_create is observable for whether it was called."""
    ctx = MagicMock()
    ctx.vector_store_manager.get_or_create.return_value = MagicMock(name="vector_store")
    ctx.auto_merging_enabled = True
    ctx.merge_threshold = 0.5
    ctx.expand_neighbors = 2
    return RAGQueryService(ctx), ctx


def _fake_folder():
    return SimpleNamespace(id=7, vector_table_uuid="uuid-7", name="f1")


_FAKE_CFG = SimpleNamespace(rag=SimpleNamespace(retrieval=SimpleNamespace(
    max_concurrent_queries=4, default_sparse_top_k=10, default_hybrid_alpha=0.5)))


async def test_acl_runs_before_any_data_access():
    """Unauthorized: verify_folder_access raises -> query_agentic raises, and never touches the vector store."""
    svc, ctx = _make_service()
    with patch(f"{_ACL}.verify_folder_access",
               side_effect=UnauthorizedAccessError(
                   resource_type="folder", resource_id="f1", reason="nope")):
        with pytest.raises(UnauthorizedAccessError):
            await svc.query_agentic(
                folder_name="f1", mode="search", query="q", file_id=None,
                top_k=5, similarity_cutoff=0.0, expand_context=False, token="tokA")
    ctx.vector_store_manager.get_or_create.assert_not_called()


async def test_list_mode_routes_and_skips_vector_store():
    """list mode: routes to handle_list, needs no vector store (should not create one)."""
    svc, ctx = _make_service()
    sentinel = {"mode": "list", "ok": True}
    with patch(f"{_ACL}.verify_folder_access", return_value=_fake_folder()), \
         patch(f"{_TH}.handle_list", return_value=sentinel) as h_list:
        out = await svc.query_agentic(
            folder_name="f1", mode="list", query="", file_id=None,
            top_k=5, similarity_cutoff=0.0, expand_context=False, token="tokA")
    assert out is sentinel
    h_list.assert_called_once()
    ctx.vector_store_manager.get_or_create.assert_not_called()


async def test_read_mode_routes_with_vector_store():
    """read mode: creates the vector store, then routes to handle_read."""
    svc, ctx = _make_service()
    sentinel = {"mode": "read"}
    with patch(f"{_ACL}.verify_folder_access", return_value=_fake_folder()), \
         patch(f"{_TH}.handle_read", new=AsyncMock(return_value=sentinel)) as h_read:
        out = await svc.query_agentic(
            folder_name="f1", mode="read", query="", file_id="file-1",
            top_k=5, similarity_cutoff=0.0, expand_context=False, token="tokA")
    assert out is sentinel
    ctx.vector_store_manager.get_or_create.assert_called_once()
    assert h_read.await_count == 1


async def test_search_mode_routes_with_retriever():
    """search mode: creates the vector store + retriever, then routes to handle_search."""
    svc, ctx = _make_service()
    sentinel = {"mode": "search"}
    with patch(f"{_ACL}.verify_folder_access", return_value=_fake_folder()), \
         patch(f"{_RQ}.Config.get_config_model", return_value=_FAKE_CFG), \
         patch(f"{_RQ}.AutoMergingRetriever", return_value=MagicMock()), \
         patch(f"{_TH}.handle_search", new=AsyncMock(return_value=sentinel)) as h_search:
        out = await svc.query_agentic(
            folder_name="f1", mode="search", query="hello", file_id=None,
            top_k=5, similarity_cutoff=0.1, expand_context=True, token="tokA")
    assert out is sentinel
    ctx.vector_store_manager.get_or_create.assert_called_once()
    assert h_search.await_count == 1


async def test_unknown_mode_returns_error():
    """Unknown mode: returns an error dict, does not raise, does not touch data."""
    svc, ctx = _make_service()
    with patch(f"{_ACL}.verify_folder_access", return_value=_fake_folder()):
        out = await svc.query_agentic(
            folder_name="f1", mode="bogus", query="", file_id=None,
            top_k=5, similarity_cutoff=0.0, expand_context=False, token="tokA")
    assert "error" in out


async def test_search_response_carries_provenance():
    """Citation provenance: MergedResult's page/headings pass through into the search response;
    results without provenance (old indexed data) do not carry these two keys."""
    from src.domain.rag.agentic_handlers import handle_search
    from src.domain.rag.auto_merging import MergedResult

    hits = [
        MergedResult(text="有溯源", score=0.9, file_id="f", file_name="a.pdf",
                     node_id="n1", node_role="leaf", chunk_index_range=[0, 0],
                     page=3, headings=["第一章"]),
        MergedResult(text="無溯源", score=0.8, file_id="f", file_name="b.txt",
                     node_id="n2", node_role="leaf", chunk_index_range=[1, 1]),
    ]
    retriever = MagicMock()
    retriever.aquery = AsyncMock(return_value=hits)
    out = await handle_search(
        query="測試查詢", folder=_fake_folder(), retriever=retriever,
        top_k=5, similarity_cutoff=0.1, expand_context=False)

    r0, r1 = out["results"]
    assert r0["page"] == 3 and r0["headings"] == ["第一章"]
    assert "page" not in r1 and "headings" not in r1
