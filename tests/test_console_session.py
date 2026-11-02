"""Admin-console login: PAT verification, session cookie, CSRF, open mode.

Tokens are signed locally (tests/conftest.py) and verified by the real
McpGateTokenVerifier with a stubbed JWKS — the same verifier the REST API uses.
"""

from __future__ import annotations

import os
import stat
import time

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.auth import console_session as cs
from src.auth.console_session import (
    COOKIE_NAME,
    ConsoleAuth,
    ConsoleIdentity,
    require_console_admin,
    require_console_session,
)

ORIGIN = "http://testserver"
CSRF = {"X-RAG-Console": "1", "Origin": ORIGIN}


def _app():
    from src.api.router.console_session import router
    app = FastAPI()
    app.include_router(router)

    @app.get("/probe")
    async def probe(ident: ConsoleIdentity = Depends(require_console_session)):
        return ident.public_view()

    @app.post("/probe")
    async def probe_post(ident: ConsoleIdentity = Depends(require_console_session)):
        return ident.public_view()

    @app.post("/admin-only")
    async def admin_only(ident: ConsoleIdentity = Depends(require_console_admin)):
        return {"ok": True, "actor": ident.actor}
    return app


@pytest.fixture()
def client(console_auth):
    return TestClient(_app())


def _login(client, token, headers=CSRF):
    return client.post("/api/console/session", json={"token": token}, headers=headers)


def _set_cookie(resp) -> str:
    return resp.headers.get("set-cookie", "")


# -- verifier error codes --------------------------------------------------------


class TestVerifierCodes:
    async def test_valid_token_carries_jti(self, gate_verifier, tokens):
        r = await gate_verifier.verify(tokens.make(jti="j-1", scopes=("a", "b")))
        assert r.valid and r.jti == "j-1" and r.scopes == ["a", "b"] and r.user_name == "alice"

    @pytest.mark.parametrize("kwargs, code", [
        ({"exp_in": -10}, "token_expired"),
        ({"aud": "http://127.0.0.1:5055/mcp"}, "wrong_audience"),
        ({"iss": "http://evil.test"}, "wrong_issuer"),
        ({"kid": "rotated"}, "unknown_signing_key"),
        ({"drop": ("exp",)}, "missing_claim"),
    ])
    async def test_rejections(self, gate_verifier, tokens, kwargs, code):
        r = await gate_verifier.verify(tokens.make(**kwargs))
        assert not r.valid and r.error_code == code

    async def test_bad_signature(self, gate_verifier, tokens):
        # Signed by another key under the trusted kid.
        r = await gate_verifier.verify(tokens.make(key=tokens.other_key))
        assert not r.valid and r.error_code == "bad_signature"

    async def test_malformed(self, gate_verifier):
        r = await gate_verifier.verify("not-a-jwt")
        assert not r.valid and r.error_code in ("malformed_token", "unknown_signing_key")

    def test_error_code_mapping_covers_the_rest(self):
        from src.auth.mcp_gate_auth import _error_code
        assert _error_code(jwt.PyJWKClientConnectionError("down")) == "jwks_unavailable"
        assert _error_code(jwt.DecodeError("x")) == "malformed_token"
        assert _error_code(jwt.ImmatureSignatureError("x")) == "invalid_token"


# -- sign in -----------------------------------------------------------------------


class TestSignIn:
    def test_valid_token_sets_session(self, client, tokens):
        tok = tokens.make(jti="jti-abcdef123", sub="alice", scopes=("rag:read",))
        r = _login(client, tok)
        assert r.status_code == 200
        d = r.json()["data"]
        assert d["mode"] == "token" and d["jti_prefix"] == "jti-abcd…"
        assert d["sub"] == "alice" and d["is_admin"] is False
        # The raw token is never echoed back.
        assert tok not in r.text
        # Session works for subsequent requests.
        me = client.get("/api/console/session").json()["data"]
        assert me["jti_prefix"] == "jti-abcd…" and me["via"] == "cookie"

    def test_cookie_flags(self, client, tokens):
        c = _set_cookie(_login(client, tokens.make())).lower()
        assert c.startswith(COOKIE_NAME + "=")
        assert "httponly" in c and "samesite=strict" in c and "path=/" in c
        assert "secure" not in c  # plain http + secure_cookie: auto
        assert "max-age=" in c

    def test_secure_flag_on_https(self, console_auth, tokens):
        c = TestClient(_app(), base_url="https://testserver")
        r = c.post("/api/console/session", json={"token": tokens.make()},
                   headers={"X-RAG-Console": "1", "Origin": "https://testserver"})
        assert r.status_code == 200 and "secure" in _set_cookie(r).lower()

    def test_secure_flag_forced_by_config(self, console_auth, tokens):
        console_auth.secure_cookie = True
        r = _login(TestClient(_app()), tokens.make())
        assert "secure" in _set_cookie(r).lower()

    def test_session_payload_never_contains_raw_token(self, client, tokens, console_auth):
        tok = tokens.make(jti="j-xyz", scopes=("rag:admin", "rag:read"))
        r = _login(client, tok)
        value = client.cookies.get(COOKIE_NAME)
        assert value and tok not in value and tok not in _set_cookie(r)
        claims = jwt.decode(value, console_auth.secret, algorithms=["HS256"])
        assert set(claims) == {"typ", "jti", "sub", "scopes", "iat", "exp"}
        assert claims["typ"] == "rag-console-session" and claims["jti"] == "j-xyz"
        assert claims["scopes"] == ["rag:admin", "rag:read"]

    def test_session_exp_is_capped_by_token_exp(self, client, tokens, console_auth):
        tok = tokens.make(exp_in=600)
        _login(client, tok)
        claims = jwt.decode(client.cookies.get(COOKIE_NAME), console_auth.secret,
                            algorithms=["HS256"])
        assert claims["exp"] == jwt.decode(tok, options={"verify_signature": False})["exp"]

    def test_session_exp_is_capped_by_session_hours(self, client, tokens, console_auth):
        console_auth.session_hours = 1
        _login(client, tokens.make(exp_in=86400 * 30))
        claims = jwt.decode(client.cookies.get(COOKIE_NAME), console_auth.secret,
                            algorithms=["HS256"])
        assert claims["exp"] - claims["iat"] == 3600

    @pytest.mark.parametrize("kwargs, code", [
        ({"exp_in": -10}, "token_expired"),
        ({"aud": "http://127.0.0.1:5055/mcp"}, "wrong_audience"),
        ({"iss": "http://evil.test"}, "wrong_issuer"),
        ({"drop": ("jti",)}, "missing_jti"),
    ])
    def test_rejected_tokens(self, client, tokens, kwargs, code):
        r = _login(client, tokens.make(**kwargs))
        assert r.status_code == 401
        assert r.json()["detail"]["error"] == code
        assert r.json()["detail"]["error_description"]
        assert COOKIE_NAME not in _set_cookie(r)

    def test_bad_signature_rejected(self, client, tokens):
        r = _login(client, tokens.make(key=tokens.other_key))
        assert r.status_code == 401 and r.json()["detail"]["error"] == "bad_signature"

    def test_empty_and_oversized_token(self, client):
        assert _login(client, "").json()["detail"]["error"] == "login_required"
        assert _login(client, "x" * (cs.MAX_TOKEN_CHARS + 1)).json()["detail"]["error"] == "malformed_token"

    def test_bearer_prefix_is_tolerated(self, client, tokens):
        assert _login(client, "Bearer " + tokens.make()).status_code == 200

    def test_jwks_unavailable_is_503(self, client, tokens, gate_verifier):
        class Down:
            def get_signing_key_from_jwt(self, token):
                raise jwt.PyJWKClientConnectionError("down")
        gate_verifier._jwk_client = Down()
        r = _login(client, tokens.make())
        assert r.status_code == 503 and r.json()["detail"]["error"] == "jwks_unavailable"

    @pytest.mark.parametrize("headers", [
        {},
        {"Origin": ORIGIN},
        {"X-RAG-Console": "1"},
        {"X-RAG-Console": "1", "Origin": "http://localhost:4568"},
    ])
    def test_login_requires_same_origin_csrf(self, client, tokens, headers):
        r = _login(client, tokens.make(), headers=headers)
        assert r.status_code == 403 and r.json()["detail"]["error"] == "csrf_failed"


# -- using the session -------------------------------------------------------------


class TestSessionUse:
    def test_no_session_is_401(self, client):
        r = client.get("/probe")
        assert r.status_code == 401 and r.json()["detail"]["error"] == "login_required"

    def test_tampered_cookie_rejected(self, client, tokens):
        _login(client, tokens.make())
        head, payload, sig = client.cookies.get(COOKIE_NAME).split(".")
        import base64
        import json
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        claims["scopes"] = ["rag:admin"]  # privilege escalation attempt
        forged = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        client.cookies.clear()
        client.cookies.set(COOKIE_NAME, f"{head}.{forged}.{sig}")
        r = client.get("/probe")
        assert r.status_code == 401 and r.json()["detail"]["error"] == "invalid_session"

    def test_cookie_signed_with_other_secret_rejected(self, client, console_auth):
        forged = jwt.encode({"typ": cs.SESSION_TYP, "jti": "victim", "sub": "x",
                             "scopes": ["rag:admin"], "iat": int(time.time()),
                             "exp": int(time.time()) + 60}, b"k" * 32, algorithm="HS256")
        client.cookies.set(COOKIE_NAME, forged)
        assert client.get("/probe").json()["detail"]["error"] == "invalid_session"

    def test_wrong_typ_rejected(self, client, console_auth):
        now = int(time.time())
        other = jwt.encode({"typ": "something-else", "jti": "j", "iat": now, "exp": now + 60},
                           console_auth.secret, algorithm="HS256")
        client.cookies.set(COOKIE_NAME, other)
        assert client.get("/probe").json()["detail"]["error"] == "invalid_session"

    def test_expired_session(self, client, tokens, console_auth):
        _login(client, tokens.make())
        console_auth.clock = lambda: time.time() + 9 * 3600
        r = client.get("/probe")
        assert r.status_code == 401 and r.json()["detail"]["error"] == "session_expired"

    def test_sign_out_clears_cookie(self, client, tokens):
        _login(client, tokens.make())
        r = client.delete("/api/console/session")
        assert r.status_code == 200
        c = _set_cookie(r).lower()
        assert c.startswith(COOKIE_NAME + "=") and ("max-age=0" in c or "expires=" in c)
        assert client.get("/probe").status_code == 401

    def test_bearer_path_for_scripts(self, client, tokens):
        # No cookie, no CSRF headers: a script just sends the PAT.
        tok = tokens.make(jti="script-jti")
        r = client.post("/probe", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        assert r.json()["via"] == "bearer" and r.json()["jti_prefix"] == "script-j…"

    def test_bad_bearer_does_not_fall_back_to_cookie(self, client, tokens):
        _login(client, tokens.make())
        r = client.get("/probe", headers={"Authorization": "Bearer junk"})
        assert r.status_code == 401

    def test_get_session_with_bearer(self, client, tokens):
        r = client.get("/api/console/session",
                       headers={"Authorization": f"Bearer {tokens.make(drop=("sub",))}"})
        assert r.status_code == 200 and r.json()["data"]["sub"] is None


class TestCsrfMatrix:
    @pytest.fixture()
    def signed_in(self, client, tokens):
        _login(client, tokens.make())
        return client

    @pytest.mark.parametrize("headers, ok", [
        ({"X-RAG-Console": "1", "Origin": ORIGIN}, True),
        ({"X-RAG-Console": "1", "Referer": ORIGIN + "/admin/folders"}, True),
        ({"X-RAG-Console": "1", "Origin": ORIGIN.upper()}, True),
        ({}, False),
        ({"Origin": ORIGIN}, False),  # header missing
        ({"X-RAG-Console": "0", "Origin": ORIGIN}, False),
        ({"X-RAG-Console": "1"}, False),  # no Origin / Referer
        ({"X-RAG-Console": "1", "Origin": "http://localhost:4568"}, False),  # same-site, other port
        ({"X-RAG-Console": "1", "Origin": "https://testserver"}, False),  # scheme differs
        ({"X-RAG-Console": "1", "Origin": "null"}, False),
        ({"X-RAG-Console": "1", "Origin": "null", "Referer": ORIGIN + "/"}, False),
        ({"X-RAG-Console": "1", "Referer": "http://evil.test/admin"}, False),
        ({"X-RAG-Console": "1", "Referer": "not a url"}, False),
    ])
    def test_unsafe_cookie_requests(self, signed_in, headers, ok):
        r = signed_in.post("/probe", headers=headers)
        assert (r.status_code == 200) is ok, r.text
        if not ok:
            assert r.status_code == 403 and r.json()["detail"]["error"] == "csrf_failed"

    def test_safe_methods_need_no_csrf(self, signed_in):
        assert signed_in.get("/probe").status_code == 200


class TestAdminGuard:
    def test_non_admin_403(self, client, tokens):
        _login(client, tokens.make(scopes=("rag:read",)))
        r = client.post("/admin-only", headers=CSRF)
        assert r.status_code == 403
        d = r.json()["detail"]
        assert d["error"] == "admin_required" and "rag:admin" in d["error_description"]

    def test_admin_allowed_and_actor_is_sub(self, client, tokens):
        _login(client, tokens.make(scopes=("rag:admin",), sub="ops"))
        r = client.post("/admin-only", headers=CSRF)
        assert r.status_code == 200 and r.json()["actor"] == "ops"

    def test_admin_scope_is_configurable(self, client, tokens, console_auth):
        console_auth.admin_scope = "rag:owner"
        _login(client, tokens.make(scopes=("rag:admin",)))
        assert client.post("/admin-only", headers=CSRF).status_code == 403


# -- open (dev) mode -----------------------------------------------------------------


class TestOpenMode:
    def test_everything_allowed_without_login(self, open_console_auth):
        c = TestClient(_app())
        me = c.get("/api/console/session").json()["data"]
        assert me["mode"] == "open" and me["is_admin"] is True
        assert c.post("/probe").status_code == 200  # no CSRF in open mode
        assert c.post("/admin-only").json()["actor"] == "console (open mode)"

    def test_session_endpoints_are_noops(self, open_console_auth):
        c = TestClient(_app())
        r = c.post("/api/console/session", json={"token": "anything"})
        assert r.status_code == 200 and r.json()["data"]["mode"] == "open"
        assert COOKIE_NAME not in _set_cookie(r)
        assert c.delete("/api/console/session").status_code == 200

    def test_startup_warning(self, make_cfg):
        from unittest.mock import patch
        with patch.object(cs.logger, "warning") as w:
            auth = cs.build_console_auth(make_cfg(enabled=False))
        assert auth.enabled is False and "DISABLED" in w.call_args.args[0]


class TestUninitialized:
    def test_guard_fails_closed(self):
        cs.set_console_auth(None)
        r = TestClient(_app()).get("/probe")
        assert r.status_code == 503 and r.json()["detail"]["error"] == "auth_not_configured"

    async def test_verifier_missing(self):
        auth = ConsoleAuth(enabled=True, verifier=None, secret=b"k" * 32)
        with pytest.raises(Exception) as e:
            await auth.verify_pat("tok")
        assert e.value.status_code == 503


# -- identity + secret -----------------------------------------------------------------


class TestIdentity:
    def test_owns_is_fail_closed(self):
        ident = ConsoleIdentity(mode="token", via="cookie", jti="a")
        assert ident.owns("a") and not ident.owns("b")
        assert not ident.owns(None) and not ident.owns("")
        assert ident.owner_key == "a"

    def test_open_owns_everything(self):
        ident = ConsoleIdentity(mode="open", via="open", is_admin=True)
        assert ident.owns(None) and ident.owns("x") and ident.owner_key is None
        assert ident.jti_prefix is None

    def test_actor_fallbacks(self):
        assert ConsoleIdentity(mode="token", via="cookie", jti="abcdefghijk").actor == "jti:abcdefgh…"
        assert ConsoleIdentity(mode="token", via="cookie", sub="bob", jti="j").actor == "bob"
        assert ConsoleIdentity(mode="token", via="cookie").actor == "console"


class TestSecret:
    def test_generated_once_and_persisted(self, tmp_path, make_cfg):
        path = tmp_path / "sub" / cs.SECRET_FILE_NAME
        a = cs.build_console_auth(make_cfg(enabled=True), secret_path=path)
        assert path.exists() and stat.S_IMODE(os.stat(path).st_mode) == 0o600
        b = cs.build_console_auth(make_cfg(enabled=True), secret_path=path)
        assert a.secret == b.secret and len(a.secret) == 32

    def test_configured_secret_wins(self, tmp_path, make_cfg):
        path = tmp_path / cs.SECRET_FILE_NAME
        a = cs.build_console_auth(make_cfg(enabled=True, session_secret="s3cret"), secret_path=path)
        b = cs.build_console_auth(make_cfg(enabled=True, session_secret="other"), secret_path=path)
        assert not path.exists() and a.secret != b.secret

    def test_blank_configured_secret_falls_back_to_file(self, tmp_path, make_cfg):
        path = tmp_path / cs.SECRET_FILE_NAME
        cs.build_console_auth(make_cfg(enabled=True, session_secret="  "), secret_path=path)
        assert path.exists()

    def test_default_path_is_under_gitignored_storage(self):
        p = cs.default_secret_path()
        assert p.parent.name == "storage" and p.name == cs.SECRET_FILE_NAME

    def test_storage_secret_is_gitignored(self):
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run(["git", "-C", root, "check-ignore", "-q",
                            f"storage/{cs.SECRET_FILE_NAME}"])
        assert r.returncode == 0

    def test_config_defaults(self):
        from src.config.model import AuthConfig
        c = AuthConfig(enabled=True).console
        assert (c.session_hours, c.session_secret, c.secure_cookie, c.admin_scope) == \
            (8, None, "auto", "rag:admin")
