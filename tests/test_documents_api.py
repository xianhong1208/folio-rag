"""REST surface of tags, metadata and document filters.

- /api/documents/search, /api/tags, /api/metadata/keys: console session or Bearer
  PAT through the real guards, always scoped to the caller's owner key;
- PATCH of a file (REST and console) validates tags / metadata (422);
- uploads (REST and console) take tags and metadata;
- POST /api/rag/query and the console playground take ``filters`` (422 on unknown
  fields / operators) and hand them to the adapter.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router.admin_settings import router as admin_router
from src.domain.rag.doc_filter import ALL_FOLDERS

ORIGIN = "http://testserver"
CSRF = {"X-RAG-Console": "1", "Origin": ORIGIN}
FID = "9e68c18e-3209-44d3-b7e9-2211b3660d66"


def _admin_app():
    app = FastAPI()
    app.include_router(admin_router)
    return app


@pytest.fixture()
def db_conn():
    """Patch the documents router's connection; the query helpers are patched per test."""
    cm = MagicMock()
    cm.__enter__.return_value = "conn"
    with patch("src.api.router.documents_api._connect", return_value=cm):
        yield


class TestDocumentsAuth:
    @pytest.mark.parametrize("method, path", [
        ("POST", "/api/documents/search"), ("GET", "/api/tags"), ("GET", "/api/metadata/keys")])
    def test_anonymous_rejected(self, console_auth, method, path):
        r = TestClient(_admin_app()).request(method, path, headers=CSRF, json={})
        assert r.status_code == 401

    def test_bearer_pat_works_without_csrf(self, console_auth, tokens, db_conn):
        tok = tokens.make(jti="j1", sub="alice", scopes=("rag:read",))
        with patch("src.domain.rag.doc_filter.search_documents", return_value=([], 0)) as s:
            r = TestClient(_admin_app()).post(
                "/api/documents/search", json={}, headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200, r.text
        assert s.call_args.kwargs["owner"] == "sub:alice"

    def test_cookie_post_needs_csrf(self, console_auth, tokens, db_conn):
        c = TestClient(_admin_app())
        tok = tokens.make(jti="j2", sub="bob", scopes=("rag:read",))
        assert c.post("/api/console/session", json={"token": tok}, headers=CSRF).status_code == 200
        with patch("src.domain.rag.doc_filter.search_documents", return_value=([], 0)) as s, \
             patch("src.domain.rag.doc_filter.tag_counts", return_value=[]):
            assert c.post("/api/documents/search", json={}).status_code == 403
            assert not s.called
            r = c.post("/api/documents/search", json={}, headers=CSRF)
            assert r.status_code == 200 and s.call_args.kwargs["owner"] == "sub:bob"
            assert c.get("/api/tags").status_code == 200        # a GET needs no CSRF header

    def test_open_mode_sees_every_folder(self, open_console_auth, db_conn):
        with patch("src.domain.rag.doc_filter.tag_counts", return_value=[]) as t:
            r = TestClient(_admin_app()).get("/api/tags")
        assert r.status_code == 200 and t.call_args.kwargs["owner"] is ALL_FOLDERS


@pytest.fixture()
def as_me(as_identity, make_identity):
    return TestClient(as_identity(_admin_app(), make_identity("sub:me")))


class TestDocumentsSearch:
    def test_page_and_next_offset(self, as_me, db_conn):
        docs = [{"id": "a"}, {"id": "b"}]
        with patch("src.domain.rag.doc_filter.search_documents", return_value=(docs, 5)) as s:
            r = as_me.post("/api/documents/search", json={
                "filters": {"tags_any": ["Invoice"], "metadata": {"year": {"gte": 2024}}},
                "sort": "name", "order": "asc", "limit": 2, "offset": 2})
        assert r.status_code == 200, r.text
        assert r.json()["data"] == {"documents": docs, "total": 5, "limit": 2, "offset": 2,
                                    "next_offset": 4}
        args, kw = s.call_args
        assert args[0] == "conn" and kw["owner"] == "sub:me"
        assert args[1].tags_any == ["Invoice"]
        assert (kw["sort"], kw["order"], kw["limit"], kw["offset"]) == ("name", "asc", 2, 2)

    def test_last_page_has_no_next(self, as_me, db_conn):
        with patch("src.domain.rag.doc_filter.search_documents", return_value=([{"id": "a"}], 1)):
            r = as_me.post("/api/documents/search", json={})
        assert r.json()["data"]["next_offset"] is None

    @pytest.mark.parametrize("body, needle", [
        ({"filters": {"tag": ["x"]}}, "tag"),
        ({"filters": {"metadata": {"year": {"gt": 1}}}}, "unknown operator"),
        ({"limit": 201}, "limit"),
        ({"sort": "random"}, "sort"),
        ({"owner": "sub:someone-else"}, "owner"),
    ])
    def test_bad_requests_are_422(self, as_me, db_conn, body, needle):
        with patch("src.domain.rag.doc_filter.search_documents") as s:
            r = as_me.post("/api/documents/search", json=body)
        assert r.status_code == 422 and needle in r.text and not s.called

    def test_tags_and_keys(self, as_me, db_conn):
        with patch("src.domain.rag.doc_filter.tag_counts",
                   return_value=[{"tag": "Invoice", "count": 2}]) as t, \
             patch("src.domain.rag.doc_filter.metadata_keys",
                   return_value=[{"key": "year", "count": 2, "types": {"number": 2}}]) as k:
            tags = as_me.get("/api/tags").json()["data"]["tags"]
            keys = as_me.get("/api/metadata/keys").json()["data"]["keys"]
        assert tags[0]["tag"] == "Invoice" and keys[0]["key"] == "year"
        assert t.call_args.kwargs["owner"] == "sub:me" == k.call_args.kwargs["owner"]


class TestConsoleFileEdits:
    def _folder(self):
        return patch("db.cached_folderdb.CachedFolderDB.get_by_id",
                     return_value=SimpleNamespace(id=7, name="f", user_token="sub:me"))

    def test_patch_normalises_and_delegates(self, as_me):
        with self._folder(), patch("src.api.router.file_api.update_file",
                                   new=AsyncMock(return_value={"id": FID})) as ep:
            r = as_me.patch(f"/api/admin/manage/folders/7/files/{FID}",
                            json={"tags": [" a ", "A", "b"], "metadata": {"year": 2024}})
        assert r.status_code == 200, r.text
        body = ep.await_args.args[2]
        assert body.tags == ["a", "b"] and body.metadata == {"year": 2024}
        assert ep.await_args.kwargs["user_token"] == "sub:me"

    @pytest.mark.parametrize("body, needle", [
        ({"metadata": {"a": {"b": 1}}}, "nested values"),
        ({"metadata": {"Bad": 1}}, "metadata key"),
        ({"tags": ["x" * 65]}, "longer than 64"),
        ({"tags": [f"t{i}" for i in range(33)]}, "at most 32 tags"),
    ])
    def test_patch_refuses_bad_values(self, as_me, body, needle):
        with self._folder(), patch("src.api.router.file_api.update_file", new=AsyncMock()) as ep:
            r = as_me.patch(f"/api/admin/manage/folders/7/files/{FID}", json=body)
        assert r.status_code == 422 and needle in r.text and not ep.await_count

    def test_upload_forwards_tags_and_metadata(self, as_me):
        with self._folder(), patch("src.api.router.file_api.upload_file",
                                   new=AsyncMock(return_value={"ok": True})) as ep:
            r = as_me.post("/api/admin/manage/folders/7/files", files={"file": ("a.txt", b"x")},
                           data={"tags": "a, b", "metadata": '{"year": 2024}'})
        assert r.status_code == 200, r.text
        assert ep.await_args.kwargs["tags"] == "a, b"
        assert ep.await_args.kwargs["metadata"] == '{"year": 2024}'

    def test_playground_passes_filters(self, as_me):
        with self._folder(), patch("src.api.router.rag_query.query_rag",
                                   new=AsyncMock(return_value={"data": {"results": []}})) as q:
            r = as_me.post("/api/admin/manage/folders/7/query",
                           json={"query": "hello", "filters": {"tags_any": ["x"]}})
        assert r.status_code == 200, r.text
        assert q.await_args.args[1].filters.tags_any == ["x"]

    def test_playground_trace_passes_filters(self, as_me):
        fake = MagicMock()
        fake.query_trace = AsyncMock(return_value={"results": []})
        with self._folder(), patch("src.adapter.rag.get_rag_adapter", return_value=fake):
            r = as_me.post("/api/admin/manage/folders/7/query",
                           json={"query": "hello", "trace": True, "filters": {"file_type": ["pdf"]}})
        assert r.status_code == 200, r.text
        assert fake.query_trace.await_args.kwargs["filters"].file_type == ["pdf"]

    def test_playground_refuses_unknown_filter_fields(self, as_me):
        with self._folder():
            r = as_me.post("/api/admin/manage/folders/7/query",
                           json={"query": "hello", "filters": {"tagz": ["x"]}})
        assert r.status_code == 422


# ---------------------------------------------------------------- REST routers

@pytest.fixture()
def rest_client():
    from src.api.dependencies.auth import extract_token
    from src.api.router import file_api, rag_query
    from src.auth.dependencies import authenticate_request
    app = FastAPI()
    app.include_router(file_api.router, prefix="/api")
    app.include_router(rag_query.router, prefix="/api/rag")
    app.dependency_overrides[authenticate_request] = lambda: {}
    app.dependency_overrides[extract_token] = lambda: "sub:me"
    return TestClient(app)


class TestRestFiles:
    def test_patch_sets_tags_and_metadata(self, rest_client):
        from datetime import datetime
        from src.adapter.model import FileConfigData
        out = FileConfigData(id=FID, folder_id=1, file_name="a", file_path="p", file_size=1,
                             tags=["a"], metadata={"y": 1}, upload_time=datetime(2024, 1, 1),
                             updated_time=datetime(2024, 1, 1))
        with patch("src.adapter.folder.FolderAdapter.get_folder", new=AsyncMock(return_value=[object()])), \
             patch("src.api.router.file_api.FileAdapter.update_file_metadata",
                   new=AsyncMock(return_value=[out])) as upd:
            r = rest_client.patch(f"/api/folders/1/files/{FID}",
                                  json={"tags": ["a", "A"], "metadata": {"y": 1}})
        assert r.status_code == 200, r.text
        assert r.json()["metadata"] == {"y": 1} and r.json()["tags"] == ["a"]
        assert upd.await_args.kwargs["tags"] == ["a"]
        assert upd.await_args.kwargs["metadata"] == {"y": 1}

    def test_patch_refuses_nested_metadata(self, rest_client):
        r = rest_client.patch(f"/api/folders/1/files/{FID}", json={"metadata": {"a": [1]}})
        assert r.status_code == 422 and "nested values" in r.text

    def test_upload_parses_tags_and_metadata(self, rest_client):
        seen = {}

        async def upload(files_data, folder_id, user_token):
            seen.update(files_data)
            raise RuntimeError("stop here")

        with patch("src.adapter.folder.FolderAdapter.get_folder", new=AsyncMock(return_value=[object()])), \
             patch("src.api.router.file_api.FileAdapter.upload_file", new=AsyncMock(side_effect=upload)):
            rest_client.post("/api/folders/1/file", files={"file": ("a.txt", b"x")},
                             data={"tags": "Invoice, invoice, 2024", "metadata": '{"year": 2024}',
                                   "auto_index": "false"})
        assert seen["tags"] == ["Invoice", "2024"] and seen["metadata"] == {"year": 2024}

    @pytest.mark.parametrize("data, needle", [
        ({"metadata": "{not json"}, "not valid JSON"),
        ({"metadata": '{"a": {"b": 1}}'}, "nested"),
        ({"tags": "x" * 65}, "longer than 64"),
    ])
    def test_upload_refuses_bad_labels_before_storing(self, rest_client, data, needle):
        with patch("src.adapter.folder.FolderAdapter.get_folder", new=AsyncMock(return_value=[object()])), \
             patch("src.api.router.file_api.FileAdapter.upload_file", new=AsyncMock()) as up:
            r = rest_client.post("/api/folders/1/file", files={"file": ("a.txt", b"x")}, data=data)
        assert r.status_code == 422 and needle in r.text and not up.await_count


class TestRestQuery:
    @pytest.fixture()
    def adapter(self):
        from src.adapter.model import RAGQueryResult
        from src.api.router import rag_query
        fake = MagicMock()
        fake.query_rag = AsyncMock(return_value=RAGQueryResult(query="q", results=[], total_results=0))
        with patch.object(rag_query, "get_rag_adapter", return_value=fake), \
             patch.object(rag_query.CachedFolderDB, "get_by_name_and_token",
                          return_value=SimpleNamespace(id=1, name="docs")), \
             patch("db.query_log_db.QueryLogDB.record"):
            yield fake

    def test_filters_reach_the_adapter(self, rest_client, adapter):
        r = rest_client.post("/api/rag/query", json={
            "query": "q?", "folder_name": "docs",
            "filters": {"tags_all": ["Signed"], "metadata": {"due": {"lte": "2024-12-31"}}}})
        assert r.status_code == 200, r.text
        flt = adapter.query_rag.await_args.kwargs["filters"]
        assert flt.tags_all == ["Signed"] and flt.meta_conditions()[0].op == "lte"

    def test_no_filters_is_none(self, rest_client, adapter):
        rest_client.post("/api/rag/query", json={"query": "q?", "folder_name": "docs"})
        assert adapter.query_rag.await_args.kwargs["filters"] is None

    @pytest.mark.parametrize("filters, needle", [
        ({"colour": "red"}, "colour"),
        ({"metadata": {"year": {"between": [1, 2]}}}, "between"),
        ({"metadata": {"k'; DROP TABLE x; --": 1}}, "metadata key"),
    ])
    def test_unknown_fields_and_operators_are_422(self, rest_client, adapter, filters, needle):
        r = rest_client.post("/api/rag/query", json={"query": "q?", "folder_name": "docs",
                                                    "filters": filters})
        assert r.status_code == 422 and needle in r.text
        assert not adapter.query_rag.await_count


class TestStoredLabels:
    def test_upload_without_tags_stores_an_empty_list_not_null(self):
        from src.adapter.file import FileAdapter
        folder = SimpleNamespace(id=7, user_token="sub:me", name="f")
        with patch("src.adapter.file.FileDB._validate_file_content",
                   return_value={"file_size": 1, "mime_type": "text/plain"}), \
             patch.object(FileAdapter, "_name_taken", return_value=False), \
             patch("src.adapter.file.FileStorage.save_file", return_value="p"), \
             patch("src.adapter.file.FileDB.create") as create:
            FileAdapter._store_file(folder, b"x", "a.txt", None, None)
            FileAdapter._store_file(folder, b"x", "b.txt", None, [" a ", "A"], {"y": 1})
        first, second = (c.kwargs for c in create.call_args_list)
        assert (first["tags"], first["doc_metadata"]) == ([], {})
        assert (second["tags"], second["doc_metadata"]) == (["a"], {"y": 1})
