
"""Admin settings API contract (feat/runtime-settings).

The service-layer logic is in test_runtime_settings.py; here we pin the HTTP
face: routing / status codes / response shape / error translation (422) /
probe guards / root redirect. The console guards are overridden with an admin
identity; the guards themselves are covered by test_console_auth_matrix.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router.admin_settings import router

_SVC = "src.adapter.runtime_settings_service"


@pytest.fixture()
def client(as_identity, make_identity):
    # Signed in as the owner of the folders the tests use ("owner-tok", with rag:admin).
    app = FastAPI()
    app.include_router(router)
    return TestClient(as_identity(app, make_identity("owner-tok", admin=True)))


class TestSettingsApi:
    def test_get_settings_shape(self, client):
        view = {"settings": [{"path": "rag.rerank.enabled", "value": True,
                              "level": "live", "overridden": False, "secret": False}]}
        with patch(f"{_SVC}.get_settings_view", return_value=view):
            r = client.get("/api/admin/settings")
        assert r.status_code == 200
        assert r.json()["data"] == view

    def test_put_applies_and_reports_warnings(self, client):
        with patch(f"{_SVC}.apply_settings",
                   return_value=({"rag.embedding.model": "m"}, ["warning text"])) as ap:
            r = client.put("/api/admin/settings",
                           json={"settings": {"rag.embedding.model": "m"}})
        assert r.status_code == 200
        body = r.json()
        assert body["data"]["applied"] == {"rag.embedding.model": "m"}
        assert body["data"]["warnings"] == ["warning text"]
        # The audit field records the session's sub (never a token)
        assert ap.call_args.kwargs.get("updated_by") == "alice"

    def test_put_with_bearer_never_records_the_token(self, client):
        with patch(f"{_SVC}.apply_settings", return_value=({}, [])) as ap:
            client.put("/api/admin/settings",
                       json={"settings": {"rag.rerank.enabled": True}},
                       headers={"Authorization": "Bearer secret-token-xyz-12345"})
        ub = str(ap.call_args.kwargs.get("updated_by"))
        assert "secret-token" not in ub and ub == "alice"

    def test_put_empty_body_422(self, client):
        assert client.put("/api/admin/settings", json={}).status_code == 422
        assert client.put("/api/admin/settings",
                          json={"settings": {}}).status_code == 422

    def test_put_validation_error_becomes_422(self, client):
        with patch(f"{_SVC}.apply_settings", side_effect=ValueError("setting path cannot be changed at runtime")):
            r = client.put("/api/admin/settings",
                           json={"settings": {"database.url": "x"}})
        assert r.status_code == 422
        assert "cannot be changed at runtime" in r.json()["detail"]

    def test_delete_resets_override(self, client):
        with patch(f"{_SVC}.reset_setting", return_value=True) as rs:
            r = client.delete("/api/admin/settings/rag.rerank.score_threshold")
        assert r.status_code == 200
        assert rs.call_args.kwargs == {"updated_by": "alice"}
        assert r.json()["data"]["reset"] == "rag.rerank.score_threshold"

    def test_delete_no_override_404(self, client):
        with patch(f"{_SVC}.reset_setting", return_value=False):
            assert client.delete(
                "/api/admin/settings/rag.rerank.score_threshold").status_code == 404


class TestProbe:
    def test_rejects_non_http_url(self, client):
        r = client.post("/api/admin/probe", json={"base_url": "ftp://x"})
        assert r.status_code == 422

    def test_reachable_lists_models(self, client):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"data": [{"id": "m1"}, {"id": "m2"}]}
        ac = MagicMock()
        ac.__aenter__ = AsyncMock(return_value=ac)
        ac.__aexit__ = AsyncMock(return_value=False)
        ac.get = AsyncMock(return_value=resp)
        with patch("src.api.router.admin_settings.httpx.AsyncClient", return_value=ac):
            r = client.post("/api/admin/probe",
                            json={"base_url": "http://svc:8000/v1", "api_key": "k"})
        d = r.json()["data"]
        assert d["reachable"] is True and d["models"] == ["m1", "m2"]

    def test_unreachable_reports_error(self, client):
        ac = MagicMock()
        ac.__aenter__ = AsyncMock(return_value=ac)
        ac.__aexit__ = AsyncMock(return_value=False)
        ac.get = AsyncMock(side_effect=OSError("conn refused at 192.0.2.5:8080"))
        with patch("src.api.router.admin_settings.httpx.AsyncClient", return_value=ac):
            r = client.post("/api/admin/probe", json={"base_url": "http://down:1/v1"})
        d = r.json()["data"]
        # Return only the exception class name -- the original message (which may contain internal addresses) must not be echoed back
        assert d["reachable"] is False and d["error"] == "OSError"
        assert "192.0.2.5" not in str(d)

    def test_probe_does_not_follow_redirects(self, client):
        resp = MagicMock(status_code=302)
        resp.json.side_effect = ValueError
        ac = MagicMock()
        ac.__aenter__ = AsyncMock(return_value=ac)
        ac.__aexit__ = AsyncMock(return_value=False)
        ac.get = AsyncMock(return_value=resp)
        with patch("src.api.router.admin_settings.httpx.AsyncClient", return_value=ac) as C:
            client.post("/api/admin/probe", json={"base_url": "http://svc:8000/v1"})
        assert C.call_args.kwargs.get("follow_redirects") is False


class TestAdminPage:
    def test_root_redirects_to_console(self, client):
        # Product front door: the root URL sends the browser to the React console
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 302 and r.headers["location"] == "/admin/"

    def test_legacy_console_removed(self, client):
        # The single-file console at /admin-classic has been deleted
        assert client.get("/admin-classic").status_code == 404
