
"""Admin settings API contract (feat/runtime-settings).

The service-layer logic is in test_runtime_settings.py; here we pin the HTTP
face: routing / status codes / response shape / error translation (422) /
probe guards / admin page reachability.
No auth (product decision): the endpoints have no auth attached, so tests call
them directly.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router.admin_settings import router

_SVC = "src.adapter.runtime_settings_service"


@pytest.fixture()
def client():
    # No auth (product decision) -- no override needed, the endpoints are callable as-is
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


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
                   return_value=({"rag.embedding.model": "m"}, ["警告文案"])) as ap:
            r = client.put("/api/admin/settings",
                           json={"settings": {"rag.embedding.model": "m"}})
        assert r.status_code == 200
        body = r.json()
        assert body["data"]["applied"] == {"rag.embedding.model": "m"}
        assert body["data"]["warnings"] == ["警告文案"]
        # No auth: without an Authorization header -> the audit field records "console"
        assert ap.call_args.kwargs.get("updated_by") == "console"

    def test_put_with_bearer_records_masked_token(self, client):
        with patch(f"{_SVC}.apply_settings", return_value=({}, [])) as ap:
            client.put("/api/admin/settings",
                       json={"settings": {"rag.rerank.enabled": True}},
                       headers={"Authorization": "Bearer secret-token-xyz-12345"})
        ub = str(ap.call_args.kwargs.get("updated_by"))
        assert "secret-token-xyz-12345" not in ub  # only the masked form is stored

    def test_put_empty_body_422(self, client):
        assert client.put("/api/admin/settings", json={}).status_code == 422
        assert client.put("/api/admin/settings",
                          json={"settings": {}}).status_code == 422

    def test_put_validation_error_becomes_422(self, client):
        with patch(f"{_SVC}.apply_settings", side_effect=ValueError("不可熱改的設定路徑")):
            r = client.put("/api/admin/settings",
                           json={"settings": {"database.url": "x"}})
        assert r.status_code == 422
        assert "不可熱改" in r.json()["detail"]

    def test_delete_resets_override(self, client):
        with patch(f"{_SVC}.reset_setting", return_value=True):
            r = client.delete("/api/admin/settings/rag.rerank.score_threshold")
        assert r.status_code == 200
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
        ac.get = AsyncMock(side_effect=OSError("conn refused at 10.0.0.5:8080"))
        with patch("src.api.router.admin_settings.httpx.AsyncClient", return_value=ac):
            r = client.post("/api/admin/probe", json={"base_url": "http://down:1/v1"})
        d = r.json()["data"]
        # Return only the exception class name -- the original message (which may contain internal addresses) must not be echoed back
        assert d["reachable"] is False and d["error"] == "OSError"
        assert "10.0.0.5" not in str(d)

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
    def test_root_serves_console(self, client):
        # Product front door: the root URL is the settings page
        r = client.get("/")
        assert r.status_code == 200 and "Control Center" in r.text

    def test_page_served(self, client):
        r = client.get("/admin")
        assert r.status_code == 200
        assert "Agentic RAG" in r.text and "Control Center" in r.text
        # External resources are limited to Google Fonts (aligned with the
        # MCP_Center design language; when offline the fallback stack degrades
        # gracefully without breaking the layout) -- everything else is inline
        import re
        externals = set(re.findall(r'https?://([^/"]+)', r.text))
        # www.w3.org is the xmlns declaration of an inline SVG, not a network request
        assert externals <= {"fonts.googleapis.com", "fonts.gstatic.com", "www.w3.org"}, externals


class TestAdminHtmlLoader:
    """Loader behavior after the console HTML was extracted into web/admin_console.html."""

    def test_reads_real_file(self):
        from src.api.router import admin_page_html as m
        html = m._load_admin_html()
        assert html.startswith("<!DOCTYPE html>") and "%%LOGO%%" in html

    def test_fallback_when_missing(self):
        from src.api.router import admin_page_html as m
        # None of the candidate paths exist -> return an explicit fallback page, no crash, no empty response
        with patch("src.utils.runtime_paths.resolve_external_dir", return_value=None), \
             patch("pathlib.Path.is_file", return_value=False):
            html = m._load_admin_html()
        assert "admin_console.html not found" in html
