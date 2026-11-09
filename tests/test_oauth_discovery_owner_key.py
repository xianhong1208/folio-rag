"""OAuth discovery fallbacks (src/api/router/oauth_discovery.py) and the owner key
(src/auth/owner_key.py) — both were below the per-file coverage floor."""

from types import SimpleNamespace
from unittest.mock import patch

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.router import oauth_discovery as od
from src.auth.owner_key import MissingSubjectError, owner_key_from_bearer


class TestOwnerKey:
    def test_sub_is_the_key(self):
        tok = jwt.encode({"jti": "abc", "sub": "u"}, "k" * 32, algorithm="HS256")
        assert owner_key_from_bearer(tok) == "sub:u"

    def test_token_without_sub_is_refused(self):
        tok = jwt.encode({"jti": "abc"}, "k" * 32, algorithm="HS256")
        with pytest.raises(MissingSubjectError):
            owner_key_from_bearer(tok)

    def test_opaque_token_is_refused(self):
        with pytest.raises(MissingSubjectError):
            owner_key_from_bearer("opaque")


_SETTINGS = SimpleNamespace(audience="http://rs/mcp", issuer="http://as",
                            required_scopes=["rag:read"])


def _app(settings=_SETTINGS):
    app = FastAPI()
    app.include_router(od.router)
    app.state.auth_settings = settings
    return TestClient(od.WellKnownCORSMiddleware(app))


class _Resp:
    def __init__(self, doc, fail=False):
        self._doc, self._fail = doc, fail

    def raise_for_status(self):
        if self._fail:
            raise RuntimeError("500")

    def json(self):
        return self._doc


class _Client:
    calls = 0
    resp = _Resp({"issuer": "http://as"})

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        _Client.calls += 1
        return _Client.resp


@pytest.fixture(autouse=True)
def _fresh():
    od.clear_cache()
    _Client.calls = 0
    _Client.resp = _Resp({"issuer": "http://as"})
    yield
    od.clear_cache()


def test_protected_resource_document():
    r = _app().get("/.well-known/oauth-protected-resource")
    assert r.status_code == 200
    assert r.json() == {"resource": "http://rs/mcp", "authorization_servers": ["http://as"],
                        "scopes_supported": ["rag:read"], "bearer_methods_supported": ["header"]}
    assert r.headers["access-control-allow-origin"] == "*"


def test_auth_disabled_is_404():
    assert _app(settings=None).get("/.well-known/oauth-protected-resource").status_code == 404


@pytest.mark.parametrize("path", ["oauth-authorization-server", "openid-configuration"])
def test_issuer_documents_are_proxied_and_cached(path):
    with patch.object(od.httpx, "AsyncClient", _Client):
        c = _app()
        assert c.get(f"/.well-known/{path}").json() == {"issuer": "http://as"}
        c.get(f"/.well-known/{path}")
    assert _Client.calls == 1


def test_unreachable_issuer_is_502():
    _Client.resp = _Resp({}, fail=True)
    with patch.object(od.httpx, "AsyncClient", _Client):
        r = _app().get("/.well-known/oauth-authorization-server")
    assert r.status_code == 502


def test_cors_preflight():
    r = _app().options("/.well-known/oauth-protected-resource")
    assert r.status_code == 204 and r.headers["access-control-allow-methods"] == "GET, OPTIONS"


def test_non_well_known_paths_untouched():
    r = _app().get("/nope")
    assert r.status_code == 404 and "access-control-allow-origin" not in r.headers
