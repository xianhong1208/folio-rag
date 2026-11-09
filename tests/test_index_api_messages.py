"""What the indexing endpoints say about themselves (QA 2026-11-09).

- Re-indexing an unchanged file returned status "already_indexed_same_content"
  with the message "File indexed successfully with 2 chunks".
- A bad folder_id on GET /indexed-files listed the same validation error twice,
  because both the endpoint and its folder dependency declared the parameter.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router.response import IndexDocumentResponse


@pytest.fixture()
def client():
    from src.api.dependencies.auth import extract_token, get_file_by_id
    from src.api.router import rag_indexing
    from src.auth.dependencies import authenticate_request

    app = FastAPI()
    app.include_router(rag_indexing.router)
    app.dependency_overrides[authenticate_request] = lambda: {}
    app.dependency_overrides[extract_token] = lambda: "tok"
    app.dependency_overrides[get_file_by_id] = lambda: SimpleNamespace(
        id=uuid4(), file_name="a.txt", file_path="p", folder_id=1, mime_type="text/plain",
        file_size=1, content_hash="h")
    return TestClient(app)


def _index(client, status):
    result = IndexDocumentResponse(file_id=str(uuid4()), index_id="i", num_chunks=2, status=status,
                                   indexed_at="", message="m")
    adapter = MagicMock(index_document=AsyncMock(return_value=result))
    with patch("src.api.router.rag_indexing.get_rag_adapter", return_value=adapter):
        return client.post(f"/file/{uuid4()}/index", json={})


def test_an_unchanged_file_is_not_reported_as_indexed(client):
    r = _index(client, "already_indexed_same_content")
    assert r.status_code == 200
    msg = r.json()["message"]
    assert "Not re-indexed" in msg and "unchanged" in msg and "successfully" not in msg


def test_a_real_index_still_reports_success(client):
    r = _index(client, "indexed")
    assert r.json()["message"] == "File indexed successfully with 2 chunks"


def test_a_bad_folder_id_is_reported_once(client):
    r = client.get("/indexed-files", params={"folder_id": "abc"})
    assert r.status_code == 422
    locs = [tuple(e["loc"]) for e in r.json()["detail"]]
    assert locs.count(("query", "folder_id")) == 1
