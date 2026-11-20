"""POST /api/search: auth (Bearer PAT or console session with CSRF), the owner scope
handed to the search, body validation and error mapping."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router.admin_settings import router as admin_router
from src.domain.rag import doc_search as DS
from src.domain.rag.doc_filter import ALL_FOLDERS

ORIGIN = "http://testserver"
CSRF = {"X-RAG-Console": "1", "Origin": ORIGIN}
OK = {"query": "q", "group_by": "passage", "results": [], "total_results": 0}


def _app():
    app = FastAPI()
    app.include_router(admin_router)
    return app


@pytest.fixture()
def adapter():
    ad = MagicMock()
    ad.search_across = AsyncMock(return_value=dict(OK))
    with patch("src.adapter.rag.get_rag_adapter", return_value=ad):
        yield ad


@pytest.fixture()
def as_me(as_identity, make_identity):
    return TestClient(as_identity(_app(), make_identity("sub:me")))


class TestAuth:
    def test_anonymous_rejected(self, console_auth, adapter):
        r = TestClient(_app()).post("/api/search", json={"query": "abc"}, headers=CSRF)
        assert r.status_code == 401 and not adapter.search_across.called

    def test_bearer_pat_scoped_to_its_person(self, console_auth, tokens, adapter):
        tok = tokens.make(jti="j1", sub="alice", scopes=("rag:read",))
        r = TestClient(_app()).post("/api/search", json={"query": "abc"},
                                    headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200, r.text
        assert adapter.search_across.await_args.kwargs["owner"] == "sub:alice"

    def test_cookie_post_needs_csrf(self, console_auth, tokens, adapter):
        c = TestClient(_app())
        tok = tokens.make(jti="j2", sub="bob", scopes=("rag:read",))
        assert c.post("/api/console/session", json={"token": tok}, headers=CSRF).status_code == 200
        assert c.post("/api/search", json={"query": "abc"}).status_code == 403
        assert not adapter.search_across.called
        r = c.post("/api/search", json={"query": "abc"}, headers=CSRF)
        assert r.status_code == 200
        assert adapter.search_across.await_args.kwargs["owner"] == "sub:bob"

    def test_open_mode_searches_every_folder(self, open_console_auth, adapter):
        r = TestClient(_app()).post("/api/search", json={"query": "abc"}, headers=CSRF)
        assert r.status_code == 200
        assert adapter.search_across.await_args.kwargs["owner"] is ALL_FOLDERS


class TestBody:
    def test_request_is_handed_over(self, as_me, adapter):
        adapter.search_across.return_value = {**OK, "group_by": "document", "documents": [],
                                              "total_results": 3}
        r = as_me.post("/api/search", json={
            "query": "  payment terms ", "folders": ["contracts", 4], "group_by": "document",
            "filters": {"tags_any": ["Signed"]}, "top_k": 7, "snippets": 2, "auto_merge": False})
        assert r.status_code == 200, r.text
        assert r.json()["message"] == "Search returned 3 document(s)"
        req = adapter.search_across.await_args.args[0]
        assert isinstance(req, DS.SearchRequest)
        assert (req.query, req.folders, req.group_by, req.top_k, req.snippets, req.auto_merge) == (
            "payment terms", ["contracts", 4], "document", 7, 2, False)
        assert req.filters.tags_any == ["Signed"]

    @pytest.mark.parametrize("body, needle", [
        ({"query": "x"}, "too short"),
        ({"query": "abc", "group_by": "file"}, "group_by"),
        ({"query": "abc", "top_k": 51}, "top_k"),
        ({"query": "abc", "snippets": 4}, "snippets"),
        ({"query": "abc", "folders": []}, "folders"),
        ({"query": "abc", "owner": "sub:someone-else"}, "owner"),
        ({"query": "abc", "filters": {"tag": ["x"]}}, "tag"),
    ])
    def test_bad_requests_are_422(self, as_me, adapter, body, needle):
        r = as_me.post("/api/search", json=body)
        assert r.status_code == 422 and needle in r.text and not adapter.search_across.called


class TestErrors:
    def test_unknown_folder_is_404(self, as_me, adapter):
        adapter.search_across.side_effect = DS.FolderNotFound("theirs")
        r = as_me.post("/api/search", json={"query": "abc", "folders": ["theirs"]})
        assert r.status_code == 404 and "theirs" in r.text

    def test_too_long_for_the_model_is_422(self, as_me, adapter):
        adapter.search_across.side_effect = RuntimeError(
            "This model's maximum context length is 512 tokens")
        r = as_me.post("/api/search", json={"query": "abc"})
        assert r.status_code == 422

    def test_other_failures_are_500(self, as_me, adapter):
        adapter.search_across.side_effect = RuntimeError("db down")
        r = as_me.post("/api/search", json={"query": "abc"})
        assert r.status_code == 500 and "db down" in r.text
