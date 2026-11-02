"""Console authorization matrix, end to end through the real guards.

Tokens are signed locally and verified by the real McpGateTokenVerifier (stub
JWKS). Folder 1 belongs to token A's jti, folder 2 to token B's. Covers:

- every /api/admin route refuses an anonymous caller (401);
- token B cannot see or touch token A's folder through any per-folder endpoint
  (404, and the underlying operation never runs), while A can;
- cross-folder aggregates are scoped to the caller's folders unless rag:admin;
- Settings / System writes need rag:admin, reads do not (except audit/resources);
- cookie sessions need the CSRF header + same origin, bearer PATs do not;
- open (auth disabled) mode allows everything.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router.admin_settings import router

ORIGIN = "http://testserver"
CSRF = {"X-RAG-Console": "1", "Origin": ORIGIN}
JTI_A, JTI_B, JTI_ADMIN = "jti-owner-a", "jti-owner-b", "jti-admin-c"
FID = "9e68c18e-3209-44d3-b7e9-2211b3660d66"

FOLDERS = {
    1: SimpleNamespace(id=1, name="docs", user_token=JTI_A, vector_table_uuid="u1"),
    2: SimpleNamespace(id=2, name="docs", user_token=JTI_B, vector_table_uuid="u2"),
    3: SimpleNamespace(id=3, name="orphan", user_token=None, vector_table_uuid="u3"),
}


def _app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture(autouse=True)
def folder_db():
    by_owner = lambda tok, use_cache=True: [f for f in FOLDERS.values() if f.user_token == tok]  # noqa: E731
    with patch("db.cached_folderdb.CachedFolderDB.get_by_id", side_effect=FOLDERS.get), \
         patch("db.cached_folderdb.CachedFolderDB.get_by_user_token", side_effect=by_owner):
        yield


def _client(console_auth, tokens, jti, scopes=("rag:read",)):
    c = TestClient(_app())
    r = c.post("/api/console/session", json={"token": tokens.make(jti=jti, scopes=scopes)},
               headers=CSRF)
    assert r.status_code == 200, r.text
    c.headers.update(CSRF)
    return c


@pytest.fixture()
def as_a(console_auth, tokens):
    return _client(console_auth, tokens, JTI_A)


@pytest.fixture()
def as_b(console_auth, tokens):
    return _client(console_auth, tokens, JTI_B)


@pytest.fixture()
def as_admin(console_auth, tokens):
    return _client(console_auth, tokens, JTI_ADMIN, scopes=("rag:read", "rag:admin"))


# -- every admin route needs a session ------------------------------------------------


def _admin_routes():
    out = []
    # The OpenAPI document lists every mounted route (FastAPI nests included routers).
    for path, ops in _app().openapi()["paths"].items():
        if not path.startswith("/api/admin"):
            continue
        concrete = re.sub(r"\{folder_id\}", "1", path)
        concrete = re.sub(r"\{file_id\}", FID, concrete)
        concrete = re.sub(r"\{[^}]+\}", "x", concrete)
        for m in ops:
            out.append((m.upper(), concrete))
    return sorted(set(out))


def test_route_inventory_is_complete():
    # If this changes, extend the matrix below (and the report table) for the new route.
    assert len(_admin_routes()) == 30


@pytest.mark.parametrize("method, path", _admin_routes())
def test_anonymous_is_rejected_everywhere(console_auth, method, path):
    c = TestClient(_app())
    r = c.request(method, path, headers=CSRF, json={})
    assert r.status_code == 401, (method, path, r.status_code)
    assert r.json()["detail"]["error"] == "login_required"


# -- per-folder isolation ---------------------------------------------------------------

# (method, path, body, delegate patched to observe whether the operation ran)
PER_FOLDER = [
    ("GET", "/api/admin/folders/{f}/files", None, "db.admin_stats.files_with_index_status"),
    ("GET", "/api/admin/folders/{f}/files/" + FID + "/chunks", None, "db.admin_stats.chunks_for_file"),
    ("GET", "/api/admin/eval/report?folder_id={f}", None, None),
    ("GET", "/api/admin/eval/history?folder_id={f}", None, None),
    ("POST", "/api/admin/eval/run", {"folder_id": "{f}"}, "src.api.router.admin_overview._load_eval_mod"),
    ("GET", "/api/admin/analytics?folder_id={f}", None, "db.query_log_db.QueryLogDB.analytics"),
    ("PATCH", "/api/admin/manage/folders/{f}", {"name": "x"}, "src.api.router.folder_api.update_folder_api"),
    ("DELETE", "/api/admin/manage/folders/{f}", None, "src.api.router.folder_api.delete_folder"),
    ("GET", "/api/admin/manage/folders/{f}/files/" + FID + "/download", None, "src.api.router.file_api.download_file"),
    ("DELETE", "/api/admin/manage/folders/{f}/files/" + FID, None, "src.api.router.file_api.delete_file"),
    ("POST", "/api/admin/manage/folders/{f}/index", {"skip_existing": True}, "src.api.router.rag_indexing.index_folder_endpoint"),
    ("POST", "/api/admin/manage/folders/{f}/index", {"skip_existing": False}, "src.api.router.rag_indexing.reindex_folder_endpoint"),
    ("POST", "/api/admin/manage/folders/{f}/rebuild-fts", {}, "src.domain.rag.vector_store_manager.rebuild_text_search_tsv"),
    ("POST", "/api/admin/manage/folders/{f}/files/" + FID + "/retry", {}, "src.adapter.rag.get_rag_adapter"),
    ("POST", "/api/admin/manage/folders/{f}/query", {"query": "hello"}, "src.api.router.rag_query.query_rag"),
    ("POST", "/api/admin/manage/folders/{f}/query/stream", {"query": "hello"}, "src.api.router.rag_query.query_rag"),
    ("POST", "/api/admin/manage/folders/{f}/jobs/j1/cancel", None, "src.api.router.rag_indexing.cancel_index_job"),
]


def _send(client, method, path, body, folder_id):
    path = path.replace("{f}", str(folder_id))
    if isinstance(body, dict):
        body = {k: (folder_id if v == "{f}" else v) for k, v in body.items()}
    return client.request(method, path, json=body)


@pytest.mark.parametrize("method, path, body, delegate", PER_FOLDER)
@pytest.mark.parametrize("folder_id", [1, 3, 999])  # A's folder, ownerless, missing
def test_other_token_cannot_reach_folder(as_b, method, path, body, delegate, folder_id):
    spy = MagicMock()
    ctx = patch(delegate, new=spy) if delegate else patch("os.path.exists", spy)
    with ctx:
        r = _send(as_b, method, path, body, folder_id)
    assert r.status_code == 404, (method, path, r.status_code, r.text)
    assert not spy.called
    assert "docs" not in r.text  # the other owner's folder name never leaks


def test_upload_to_foreign_folder_is_404(as_b):
    with patch("src.api.router.file_api.upload_file", new=AsyncMock()) as ep:
        r = as_b.post("/api/admin/manage/folders/1/files", files={"file": ("a.txt", b"x")})
    assert r.status_code == 404 and not ep.await_count


@pytest.mark.parametrize("method, path, body, delegate", [
    p for p in PER_FOLDER
    if p[3] and p[3].startswith(("src.api.router.folder_api", "src.api.router.file_api",
                                 "src.api.router.rag_indexing", "db.admin_stats",
                                 "db.query_log_db"))
])
def test_owner_reaches_own_folder(as_a, method, path, body, delegate):
    ret = {"data": {}, "message": "ok"}
    mock = AsyncMock(return_value=ret) if not delegate.startswith("db.") else MagicMock(return_value=ret)
    with patch(delegate, new=mock):
        r = _send(as_a, method, path, body, 1)
    assert r.status_code == 200, r.text
    assert mock.called
    # Operations act with the session's own owner key (jti A).
    kwargs = mock.call_args.kwargs
    token = kwargs.get("user_token") or kwargs.get("token")
    if token is not None:
        assert token == JTI_A


def test_owner_query_uses_own_jti(as_a):
    with patch("src.api.router.rag_query.query_rag",
               new=AsyncMock(return_value={"data": {"results": []}})) as q:
        r = as_a.post("/api/admin/manage/folders/1/query", json={"query": "hello"})
    assert r.status_code == 200 and q.await_args.kwargs["token"] == JTI_A


def test_admin_scope_does_not_grant_other_owners_folders(as_admin):
    with patch("src.api.router.folder_api.delete_folder", new=AsyncMock()) as ep:
        r = as_admin.delete("/api/admin/manage/folders/1")
    assert r.status_code == 404 and not ep.await_count


def test_created_folder_belongs_to_session(as_b):
    with patch("src.api.router.folder_api.create_folder",
               new=AsyncMock(return_value={"id": 5})) as ep:
        r = as_b.post("/api/admin/manage/folders", json={"name": "n", "owner_token": JTI_A})
    assert r.status_code == 200 and ep.await_args.kwargs["user_token"] == JTI_B


# -- aggregates -------------------------------------------------------------------------


class TestAggregates:
    def test_folder_list_is_own_only(self, as_b, as_admin):
        with patch("db.admin_stats.folders_with_index_stats", return_value=[]) as m:
            as_b.get("/api/admin/folders")
            as_admin.get("/api/admin/folders")
        assert [c.kwargs["owner"] for c in m.call_args_list] == [JTI_B, JTI_ADMIN]

    def _overview(self, client):
        base = {"folders": 0, "files": 0, "total_size_bytes": 0, "index": {}, "jobs": {}}
        with patch("db.admin_stats.overview", return_value=dict(base)) as ov, \
             patch("db.admin_stats.indexing_trend", return_value=[]) as tr, \
             patch("src.config.runtime_overrides.get_effective", return_value={}), \
             patch("src.config.config_manager.Config.get_config_model",
                   return_value=SimpleNamespace()):
            r = client.get("/api/admin/overview")
        return r, ov.call_args.kwargs["owner"], tr.call_args.kwargs["owner"]

    def test_overview_scoped_for_non_admin(self, as_b):
        r, owner, trend_owner = self._overview(as_b)
        assert (owner, trend_owner) == (JTI_B, JTI_B) and r.json()["data"]["scope"] == "own"

    def test_overview_global_for_admin(self, as_admin):
        r, owner, _ = self._overview(as_admin)
        assert owner is None and r.json()["data"]["scope"] == "all"

    def test_jobs(self, as_b, as_admin):
        with patch("db.admin_stats.recent_jobs", return_value=[]) as m:
            as_b.get("/api/admin/jobs")
            as_admin.get("/api/admin/jobs")
        assert m.call_args_list[0].kwargs == {"owner": JTI_B, "viewer": JTI_B}
        assert m.call_args_list[1].kwargs == {"owner": None, "viewer": JTI_ADMIN}

    def test_analytics_without_folder(self, as_b, as_admin):
        with patch("db.query_log_db.QueryLogDB.analytics", return_value={}) as m:
            as_b.get("/api/admin/analytics")
            as_admin.get("/api/admin/analytics")
        assert m.call_args_list[0].kwargs["folder_ids"] == {2}
        assert m.call_args_list[1].kwargs["folder_ids"] is None

    def test_active_stages_filtered(self, as_b):
        from src.domain.rag.index_job_manager import IndexingJobManager
        mgr = MagicMock()
        mgr.active_file_stages.return_value = {
            "fa": {"folder_id": 1, "stage": "embedding"},
            "fb": {"folder_id": 2, "stage": "parsing"},
        }
        with patch.object(IndexingJobManager, "get_instance", return_value=mgr):
            stages = as_b.get("/api/admin/active-stages").json()["data"]["stages"]
        assert list(stages) == ["fb"]

    def test_eval_status_hides_other_owners_run(self, as_a, as_b):
        from src.api.router import admin_overview as ov
        ov._EVAL.update(running=True, folder="1", done=3, total=10,
                        result={"secret": "questions"}, error=None, started_at=1.0, owner=JTI_A)
        try:
            mine = as_a.get("/api/admin/eval/status").json()["data"]
            theirs = as_b.get("/api/admin/eval/status").json()["data"]
        finally:
            ov._EVAL.update(running=False, folder=None, done=0, total=0, result=None,
                            error=None, started_at=None, owner=None)
        assert mine["folder"] == "1" and mine["result"] == {"secret": "questions"}
        assert "owner" not in mine
        assert theirs["running"] is True and theirs["busy"] is True
        assert theirs["folder"] is None and theirs["result"] is None and "owner" not in theirs


# -- admin tier --------------------------------------------------------------------------

_SVC = "src.adapter.runtime_settings_service"

ADMIN_WRITES = [
    ("PUT", "/api/admin/settings", {"settings": {"rag.rerank.enabled": True}}, f"{_SVC}.apply_settings", ({}, [])),
    ("DELETE", "/api/admin/settings/rag.rerank.enabled", None, f"{_SVC}.reset_setting", True),
    ("POST", "/api/admin/probe", {"base_url": "http://svc/v1"}, "src.api.router.admin_settings.httpx.AsyncClient", None),
    ("GET", "/api/admin/audit", None, f"{_SVC}.audit_log", []),
    ("GET", "/api/admin/resources", None, "db.admin_stats.system_resources", {}),
]


@pytest.mark.parametrize("method, path, body, target, ret", ADMIN_WRITES)
def test_admin_only_endpoints_refuse_non_admin(as_b, method, path, body, target, ret):
    with patch(target, return_value=ret) as m:
        r = as_b.request(method, path, json=body)
    assert r.status_code == 403 and r.json()["detail"]["error"] == "admin_required"
    assert not m.called


@pytest.mark.parametrize("method, path, body, target, ret", ADMIN_WRITES[:2] + ADMIN_WRITES[3:])
def test_admin_only_endpoints_allow_admin(as_admin, method, path, body, target, ret):
    with patch(target, return_value=ret) as m:
        r = as_admin.request(method, path, json=body)
    assert r.status_code == 200 and m.called
    if target.endswith(("apply_settings", "reset_setting")):
        assert m.call_args.kwargs["updated_by"] == "alice"  # the token's sub


def test_non_admin_can_read_settings_and_health(as_b):
    with patch(f"{_SVC}.get_settings_view", return_value={"settings": []}):
        assert as_b.get("/api/admin/settings").status_code == 200
    with patch("src.api.router.admin_overview._probe_db_health", return_value=("ok", "x", 1)), \
         patch("src.config.runtime_overrides.get_effective", return_value={}), \
         patch("src.config.config_manager.Config.get_config_model",
               return_value=SimpleNamespace(database=SimpleNamespace(url=""))):
        assert as_b.get("/api/admin/health").status_code == 200


# -- CSRF / bearer / open mode -------------------------------------------------------------


def test_cookie_write_without_csrf_header_is_403(as_a):
    as_a.headers.pop("X-RAG-Console")
    with patch("src.api.router.folder_api.delete_folder", new=AsyncMock()) as ep:
        r = as_a.delete("/api/admin/manage/folders/1")
    assert r.status_code == 403 and not ep.await_count


def test_cookie_write_from_other_origin_is_403(as_a):
    as_a.headers["Origin"] = "http://localhost:4568"
    with patch("src.api.router.folder_api.delete_folder", new=AsyncMock()) as ep:
        r = as_a.delete("/api/admin/manage/folders/1")
    assert r.status_code == 403 and not ep.await_count


def test_bearer_script_needs_no_csrf_and_is_scoped(console_auth, tokens):
    c = TestClient(_app())
    auth = {"Authorization": f"Bearer {tokens.make(jti=JTI_B)}"}
    with patch("src.api.router.folder_api.delete_folder", new=AsyncMock(return_value={})) as ep:
        assert c.delete("/api/admin/manage/folders/1", headers=auth).status_code == 404
        assert c.delete("/api/admin/manage/folders/2", headers=auth).status_code == 200
    assert ep.await_args.kwargs["user_token"] == JTI_B


def test_bearer_non_admin_cannot_change_settings(console_auth, tokens):
    c = TestClient(_app())
    r = c.put("/api/admin/settings", json={"settings": {"x": 1}},
              headers={"Authorization": f"Bearer {tokens.make(jti=JTI_B)}"})
    assert r.status_code == 403


def test_open_mode_allows_everything(open_console_auth):
    c = TestClient(_app())
    with patch("db.admin_stats.folders_with_index_stats", return_value=[]) as m:
        assert c.get("/api/admin/folders").status_code == 200
    assert m.call_args.kwargs["owner"] is None
    with patch(f"{_SVC}.apply_settings", return_value=({}, [])) as ap:
        assert c.put("/api/admin/settings", json={"settings": {"x": 1}}).status_code == 200
    assert ap.call_args.kwargs["updated_by"] == "console (open mode)"
    with patch("src.api.router.folder_api.delete_folder", new=AsyncMock(return_value={})) as ep:
        assert c.delete("/api/admin/manage/folders/1").status_code == 200
    assert ep.await_args.kwargs["user_token"] == JTI_A


def test_root_redirect_is_public(console_auth):
    r = TestClient(_app()).get("/", follow_redirects=False)
    assert r.status_code == 302
