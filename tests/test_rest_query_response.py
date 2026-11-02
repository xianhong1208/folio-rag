"""POST /query must serialize real query results through its response schema.

The domain DTO types node_id as a UUID while the API schema declares a string, so
returning a python-mode dump failed FastAPI's response validation (a 500) on every
query that found anything. retrieval_time_ms was also declared but never filled.
"""
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.adapter.model import RAGQueryResult
from src.api.dependencies.auth import extract_token
from src.api.router import rag_query
from src.auth.dependencies import authenticate_request
from src.domain.rag.dto import RAGChunkMetadata, RAGSearchResult

_NODE_ID = uuid.uuid4()


def _result():
    hit = RAGSearchResult(
        text="chunk text", score=0.91,
        metadata=RAGChunkMetadata(node_id=_NODE_ID, mcp_file_id="f-1",
                                  file_name="a.pdf", folder_name="docs", page=3))
    return RAGQueryResult(query="q?", results=[hit], total_results=1)


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(rag_query.router)
    app.dependency_overrides[authenticate_request] = lambda: {}
    app.dependency_overrides[extract_token] = lambda: "owner-key"
    return TestClient(app)


@pytest.fixture()
def adapter():
    fake = MagicMock()
    fake.query_rag = AsyncMock(return_value=_result())
    with patch.object(rag_query, "get_rag_adapter", return_value=fake), \
         patch.object(rag_query.CachedFolderDB, "get_by_name_and_token",
                      return_value=SimpleNamespace(id=1, name="docs")), \
         patch("db.query_log_db.QueryLogDB.record"):
        yield fake


def test_results_with_uuid_node_id_pass_the_response_schema(client, adapter):
    r = client.post("/query", json={"query": "q?", "folder_name": "docs"})
    assert r.status_code == 200, r.text
    hit = r.json()["data"]["results"][0]
    assert hit["metadata"]["node_id"] == str(_NODE_ID)
    assert hit["metadata"]["page"] == 3


def test_retrieval_time_ms_is_populated(client, adapter):
    r = client.post("/query", json={"query": "q?", "folder_name": "docs"})
    ms = r.json()["data"]["retrieval_time_ms"]
    assert isinstance(ms, (int, float)) and ms >= 0


async def test_direct_call_returns_json_types_and_leaves_cached_result_untouched(adapter):
    """The console calls query_rag as a function (no response validation)."""
    cached = adapter.query_rag.return_value
    req = rag_query.QueryRequest(query="q?", folder_name="docs")
    resp = await rag_query.query_rag(MagicMock(), req, token="owner-key")
    assert resp["data"]["results"][0]["metadata"]["node_id"] == str(_NODE_ID)
    assert resp["data"]["retrieval_time_ms"] is not None
    assert cached.retrieval_time_ms is None
