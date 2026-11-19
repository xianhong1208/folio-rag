"""MCP Gate revocation feed: the poller, its configuration, and every verification path.

The feed is faked with httpx.MockTransport (no network); tokens are signed locally
(tests/conftest.py) and verified by the real verifiers.
"""

from __future__ import annotations

import asyncio
import time
import urllib.parse
from types import SimpleNamespace

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.auth import revocation as rv
from src.auth.revocation import (
    FAIL_CLOSED,
    FAIL_OPEN,
    REVOKED,
    UNAVAILABLE,
    RevocationFeedError,
    RevocationList,
    RevocationSettings,
)
from tests.conftest import AUDIENCE, ISSUER

ORIGIN = "http://testserver"
CSRF = {"X-RAG-Console": "1", "Origin": ORIGIN}


class FakeFeed:
    """A scripted /oauth/revoked endpoint that records every request."""

    def __init__(self):
        self.requests: list[dict] = []
        self.responses: list = []  # httpx.Response | Exception | callable(form) -> Response
        self.now = 1_000_000

    def queue(self, revoked=(), now=None, status=200, body=None):
        payload = body if body is not None else {
            "revoked": [{"jti": j, "revoked_at": self.now, "exp": e} for j, e in revoked],
            "now": now if now is not None else self.now,
        }
        self.responses.append(httpx.Response(status, json=payload))

    def handler(self, request: httpx.Request) -> httpx.Response:
        form = dict(urllib.parse.parse_qsl(request.content.decode()))
        self.requests.append({"url": str(request.url), "form": form})
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _settings(**kw):
    base = dict(issuer="http://gate.test/", client_id="rs-1", client_secret="s3cret",
                poll_interval=30.0, fail_mode=FAIL_OPEN, stale_after_intervals=4)
    base.update(kw)
    return RevocationSettings(**base)


def _rl(feed, clock=None, rng=lambda a, b: 0.0, **kw):
    return RevocationList(_settings(**kw), http_client=feed.client(),
                          clock=clock or Clock(), rng=rng)


@pytest.fixture(autouse=True)
def _no_global_list():
    rv.set_revocation_list(None)
    yield
    rv.set_revocation_list(None)


# -- the poller ---------------------------------------------------------------------


async def test_first_poll_is_full_then_incremental_with_overlap():
    feed, clock = FakeFeed(), Clock()
    rl = _rl(feed, clock)
    feed.queue([("a", 1_000_500)], now=1_000_000)
    feed.queue([("b", 1_000_900)], now=1_000_030)

    assert await rl.poll_once() == 1
    clock.t += 30
    assert await rl.poll_once() == 1

    first, second = feed.requests
    assert first["url"] == "http://gate.test/oauth/revoked"
    assert first["form"] == {"client_id": "rs-1", "client_secret": "s3cret"}
    # since = MCP Gate's own clock minus the overlap, not ours
    assert second["form"]["since"] == str(1_000_000 - rv.SINCE_OVERLAP_SEC)
    assert "a" in rl and "b" in rl and len(rl) == 2


async def test_periodic_full_resync_replaces_the_list():
    feed, clock = FakeFeed(), Clock()
    rl = _rl(feed, clock)
    feed.queue([("a", 2_000_000), ("b", 2_000_000)])
    await rl.poll_once()
    clock.t += rv.FULL_RESYNC_SEC
    feed.queue([("b", 2_000_000)])
    await rl.poll_once()
    assert "since" not in feed.requests[1]["form"]
    assert "a" not in rl and "b" in rl


async def test_entries_are_pruned_after_exp():
    feed, clock = FakeFeed(), Clock()
    rl = _rl(feed, clock)
    feed.queue([("short", 1_000_010), ("long", 1_009_999)], now=1_000_000)
    await rl.poll_once()
    assert "short" in rl
    # still inside the skew grace
    clock.t = 1_000_010 + rv.PRUNE_GRACE_SEC
    feed.queue([], now=int(clock.t))
    await rl.poll_once()
    assert "short" in rl
    clock.t += 1 + rv.PRUNE_GRACE_SEC
    feed.queue([], now=int(clock.t))
    await rl.poll_once()
    assert "short" not in rl and "long" in rl


async def test_malformed_items_are_skipped_and_missing_exp_is_kept():
    feed = FakeFeed()
    rl = _rl(feed)
    feed.queue(body={"revoked": [{"jti": 5}, "junk", {"jti": "noexp"}, {"jti": "ok", "exp": "x"}],
                     "now": "bad"})
    assert await rl.poll_once() == 2
    assert "noexp" in rl and "ok" in rl and 5 not in rl


@pytest.mark.parametrize("response, match", [
    (httpx.Response(401, json={"error": "invalid_client", "error_description": "client authentication failed"}),
     "HTTP 401: invalid_client \\(client authentication failed\\)"),
    (httpx.Response(403, json={"error": "unauthorized_client"}), "HTTP 403: unauthorized_client"),
    (httpx.Response(502, text="bad gateway"), "HTTP 502: bad gateway"),
    (httpx.Response(200, text="not json"), "not JSON"),
    (httpx.Response(200, json={"items": []}), "no 'revoked' list"),
])
async def test_feed_errors_raise(response, match):
    feed = FakeFeed()
    feed.responses.append(response)
    rl = _rl(feed)
    with pytest.raises(RevocationFeedError, match=match):
        await rl.poll_once()
    assert rl.status()["last_success"] is None


async def test_run_loop_backs_off_on_errors_and_recovers():
    feed, clock = FakeFeed(), Clock()
    delays: list[float] = []

    async def fake_sleep(d):
        delays.append(d)
        clock.t += d
        if len(delays) == 7:
            raise asyncio.CancelledError

    rl = RevocationList(_settings(), http_client=feed.client(), clock=clock,
                        sleep=fake_sleep, rng=lambda a, b: 0.0)
    feed.queue([("a", 9_999_999)])
    for _ in range(5):
        feed.responses.append(httpx.ConnectError("refused"))
    feed.queue([("b", 9_999_999)])

    with pytest.raises(asyncio.CancelledError):
        await rl.run()
    # ok, then 5 failures doubling up to the cap, then back to the interval
    assert delays == [30, 30, 60, 120, 240, 300, 30]
    assert "a" in rl and "b" in rl
    st = rl.status()
    assert st["consecutive_failures"] == 0 and st["last_error"] is None


def test_jitter_stays_within_ten_percent():
    rl = RevocationList(_settings(), http_client=httpx.AsyncClient(), rng=lambda a, b: b)
    assert rl.next_delay() == pytest.approx(33.0)
    rl = RevocationList(_settings(), http_client=httpx.AsyncClient(), rng=lambda a, b: a)
    assert rl.next_delay() == pytest.approx(27.0)
    rl = RevocationList(_settings(poll_interval=600), http_client=httpx.AsyncClient(), rng=lambda a, b: 0)
    rl._failures = 10
    assert rl.next_delay() == 600  # the cap never undercuts the interval


async def test_fail_open_keeps_accepting_when_the_feed_is_down():
    feed, clock = FakeFeed(), Clock()
    rl = _rl(feed, clock, fail_mode=FAIL_OPEN)
    assert rl.check("x") is None  # nothing fetched yet: accepted
    feed.queue([("gone", 9_999_999)])
    await rl.poll_once()
    clock.t += 10_000
    feed.responses.append(httpx.ConnectError("down"))
    with pytest.raises(httpx.ConnectError):
        await rl.poll_once()
    rl._record_failure(httpx.ConnectError("down"))
    assert rl.is_stale()
    assert rl.check("x") is None
    assert rl.check("gone") == REVOKED  # what we already know still applies
    assert rl.status()["status"] == "degraded"


async def test_fail_closed_rejects_once_stale():
    feed, clock = FakeFeed(), Clock()
    rl = _rl(feed, clock, fail_mode=FAIL_CLOSED)
    assert rl.check("x") == UNAVAILABLE  # before the first successful poll
    feed.queue([])
    await rl.poll_once()
    assert rl.check("x") is None
    clock.t += 4 * 30  # exactly stale_after: still fresh
    assert rl.check("x") is None
    clock.t += 1
    assert rl.check("x") == UNAVAILABLE
    rl._record_failure(RuntimeError("down"))
    assert rl.status()["stale"] is True
    feed.queue([])
    await rl.poll_once()
    assert rl.check("x") is None


def test_check_ignores_missing_jti_and_non_strings():
    rl = RevocationList(_settings(), http_client=httpx.AsyncClient())
    rl._revoked = {"a": 1}
    assert rl.check(None) is None
    assert 1 not in rl
    assert rl.check("a") == REVOKED


# -- lifecycle ----------------------------------------------------------------------


async def test_start_stop_lifecycle():
    feed = FakeFeed()
    for _ in range(3):
        feed.queue([("a", 9_999_999_999)], now=int(time.time()))
    rl = RevocationList(_settings(poll_interval=3600), http_client=feed.client())
    rl.start()
    task = rl._task
    rl.start()  # idempotent
    assert rl._task is task and rl.running
    for _ in range(50):
        if "a" in rl:
            break
        await asyncio.sleep(0.01)
    assert "a" in rl
    await rl.stop()
    assert not rl.running and task.cancelled()
    await rl.stop()  # idempotent


async def test_owned_client_is_closed_on_stop_and_recreated_on_start(monkeypatch):
    rl = RevocationList(_settings())
    first = rl._http

    async def never(self):
        await asyncio.sleep(3600)
    monkeypatch.setattr(RevocationList, "run", never)
    rl.start()
    await rl.stop()
    assert first.is_closed
    rl.start()
    assert rl._http is not first and not rl._http.is_closed
    await rl.stop()


async def test_revocation_lifespan_starts_and_stops_the_global_list(monkeypatch):
    started = []

    async def never(self):
        started.append(self)
        await asyncio.sleep(3600)
    monkeypatch.setattr(RevocationList, "run", never)
    rl = RevocationList(_settings(), http_client=httpx.AsyncClient())
    rv.set_revocation_list(rl)
    async with rv.revocation_lifespan():
        await asyncio.sleep(0)
        assert rl.running and started == [rl]
    assert not rl.running


async def test_revocation_lifespan_is_a_noop_when_disabled():
    async with rv.revocation_lifespan():
        pass
    assert rv.revocation_status() == {"status": "disabled", "enabled": False}
    assert rv.check_revoked("anything") is None


# -- configuration ------------------------------------------------------------------


def _config(auth_enabled=True, **rev):
    from src.config.model import RevocationConfig
    return SimpleNamespace(auth=SimpleNamespace(enabled=auth_enabled,
                                                revocation=RevocationConfig(**rev)))


REFS = {"auth.revocation.client_secret": "FOLIO_REVOCATION_CLIENT_SECRET"}


def test_build_disabled_by_default():
    assert rv.build_revocation_list(_config(), issuer=ISSUER, env_refs=REFS) is None
    assert rv.build_revocation_list(SimpleNamespace(), issuer=ISSUER) is None


def test_build_ignored_when_auth_disabled():
    cfg = _config(auth_enabled=False, enabled=True, client_id="c", client_secret="s")
    assert rv.build_revocation_list(cfg, issuer=None, env_refs=REFS) is None


@pytest.mark.parametrize("rev", [dict(client_id="c"), dict(client_secret="s"),
                                 dict(client_id="", client_secret="")])
def test_build_requires_credentials(rev):
    with pytest.raises(RuntimeError, match="client_id / client_secret"):
        rv.build_revocation_list(_config(enabled=True, **rev), issuer=ISSUER, env_refs=REFS)


def test_build_rejects_a_literal_secret():
    cfg = _config(enabled=True, client_id="c", client_secret="literal")
    with pytest.raises(RuntimeError, match="ENV_VAR"):
        rv.build_revocation_list(cfg, issuer=ISSUER, env_refs={})


def test_build_rejects_unknown_fail_mode():
    cfg = SimpleNamespace(auth=SimpleNamespace(enabled=True, revocation=SimpleNamespace(
        enabled=True, client_id="c", client_secret="s", fail_mode="maybe")))
    with pytest.raises(RuntimeError, match="fail_mode"):
        rv.build_revocation_list(cfg, issuer=ISSUER, env_refs=REFS)


async def test_build_resolves_settings():
    cfg = _config(enabled=True, client_id=" c ", client_secret="s", poll_interval_sec=15,
                  fail_mode="closed", stale_after_intervals=3)
    rl = rv.build_revocation_list(cfg, issuer=ISSUER + "/", env_refs=REFS)
    s = rl.settings
    assert (s.feed_url, s.client_id, s.poll_interval, s.fail_mode, s.stale_after) == (
        f"{ISSUER}/oauth/revoked", "c", 15.0, "closed", 45.0)
    assert "s'" not in repr(s)  # the secret is not in the repr
    await rl.stop()


def test_config_model_defaults_and_validation():
    from pydantic import ValidationError
    from src.config.model import AuthConfig, RevocationConfig
    rc = AuthConfig(enabled=True).revocation
    assert (rc.enabled, rc.poll_interval_sec, rc.fail_mode, rc.stale_after_intervals) == (
        False, 30, "open", 4)
    with pytest.raises(ValidationError):
        RevocationConfig(fail_mode="sometimes")
    with pytest.raises(ValidationError):
        RevocationConfig(poll_interval_sec=1)


def test_init_auth_installs_and_clears_the_list(monkeypatch):
    from src.auth import init_auth
    from src.config.config_manager import Config
    monkeypatch.setattr(Config, "get_env_refs", classmethod(lambda cls: dict(REFS)))
    monkeypatch.setattr(init_auth, "McpGateTokenVerifier", lambda settings: object())
    monkeypatch.setattr(init_auth, "set_token_verifier", lambda v: None)
    monkeypatch.setattr(init_auth, "set_console_auth", lambda a: None)
    monkeypatch.setattr(init_auth, "build_console_auth", lambda *a, **k: None)
    from src.config.model import RevocationConfig
    cfg = SimpleNamespace(
        auth=SimpleNamespace(enabled=True, issuer=ISSUER, audience=AUDIENCE, required_scopes=[],
                             revocation=RevocationConfig(enabled=True, client_id="c",
                                                         client_secret="s")),
        server=SimpleNamespace(host="127.0.0.1", port=5039))
    init_auth.initialize_auth_system(cfg)
    assert rv.get_revocation_list().settings.client_id == "c"
    cfg.auth.revocation = RevocationConfig()
    init_auth.initialize_auth_system(cfg)
    assert rv.get_revocation_list() is None
    rv.set_revocation_list(RevocationList(_settings(), http_client=httpx.AsyncClient()))
    cfg.auth.enabled = False
    init_auth.initialize_auth_system(cfg)
    assert rv.get_revocation_list() is None


def test_auth_yaml_template_ships_disabled_with_env_refs():
    import yaml
    from src.config.profile import env_refs
    raw = yaml.safe_load(open("config/auth.yaml", encoding="utf-8"))
    rev = raw["auth"]["revocation"]
    refs = env_refs(raw)
    assert refs["auth.revocation.client_secret"] == "FOLIO_REVOCATION_CLIENT_SECRET"
    assert refs["auth.revocation.client_id"] == "FOLIO_REVOCATION_CLIENT_ID"
    assert rev["enabled"].endswith(":-false}")
    assert rev["fail_mode"] == "open"


# -- verification paths -------------------------------------------------------------


def _install(revoked=(), fail_mode=FAIL_OPEN, fresh=True):
    rl = RevocationList(_settings(fail_mode=fail_mode), http_client=httpx.AsyncClient())
    rl._revoked = {j: 9_999_999_999 for j in revoked}
    if fresh:
        rl._last_success = time.time()
    rv.set_revocation_list(rl)
    return rl


async def test_rest_verifier_rejects_a_revoked_jti(gate_verifier, tokens):
    good, bad = tokens.make(jti="keep"), tokens.make(jti="gone")
    _install(revoked={"gone"})
    assert (await gate_verifier.verify(good)).valid
    res = await gate_verifier.verify(bad)
    assert not res.valid and res.error_code == REVOKED and res.error == "token has been revoked"


async def test_rest_verifier_fail_closed(gate_verifier, tokens):
    _install(fail_mode=FAIL_CLOSED, fresh=False)
    res = await gate_verifier.verify(tokens.make())
    assert not res.valid and res.error_code == UNAVAILABLE


async def test_rest_verifier_keeps_other_failures(gate_verifier, tokens):
    _install(revoked={"gone"})
    res = await gate_verifier.verify(tokens.make(jti="gone", exp_in=-10))
    assert res.error_code == "token_expired"


def test_rest_dependency_answers_401(gate_verifier, tokens):
    from src.auth.dependencies import authenticate_request, set_token_verifier
    app = FastAPI()

    @app.get("/x")
    async def x(user=Depends(authenticate_request)):
        return user
    set_token_verifier(gate_verifier)
    try:
        _install(revoked={"gone"})
        c = TestClient(app)
        assert c.get("/x", headers={"Authorization": f"Bearer {tokens.make()}"}).status_code == 200
        r = c.get("/x", headers={"Authorization": f"Bearer {tokens.make(jti='gone')}"})
        assert r.status_code == 401 and r.json()["detail"]["error"] == "token has been revoked"
    finally:
        set_token_verifier(None)


def _mcp_verifier(rsa_key):
    from src.auth.fastmcp_verifier import RevocationAwareJWTVerifier
    pem = rsa_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return RevocationAwareJWTVerifier(public_key=pem, issuer=ISSUER, audience=AUDIENCE)


async def test_mcp_verifier_rejects_a_revoked_jti(rsa_key, tokens):
    v = _mcp_verifier(rsa_key)
    _install(revoked={"gone"})
    assert await v.verify_token(tokens.make(jti="keep")) is not None
    assert await v.verify_token(tokens.make(jti="gone")) is None
    assert await v.verify_token("garbage") is None


async def test_mcp_verifier_fail_closed(rsa_key, tokens):
    v = _mcp_verifier(rsa_key)
    _install(fail_mode=FAIL_CLOSED, fresh=False)
    assert await v.verify_token(tokens.make()) is None
    rv.set_revocation_list(None)
    assert await v.verify_token(tokens.make()) is not None


def test_auth_provider_uses_the_revocation_aware_verifier():
    from src.auth.fastmcp_verifier import RevocationAwareJWTVerifier
    from src.auth.mcp_gate_auth import build_auth_provider
    cfg = SimpleNamespace(
        auth=SimpleNamespace(enabled=True, issuer=ISSUER, audience=AUDIENCE, required_scopes=[]),
        server=SimpleNamespace(host="127.0.0.1", port=5039))
    assert isinstance(build_auth_provider(cfg).token_verifier, RevocationAwareJWTVerifier)


async def test_mcp_transport_answers_401_invalid_token(rsa_key, tokens):
    from fastmcp import FastMCP
    from fastmcp.server.auth import RemoteAuthProvider
    provider = RemoteAuthProvider(token_verifier=_mcp_verifier(rsa_key),
                                  authorization_servers=[ISSUER], base_url="http://127.0.0.1:5039")
    mcp = FastMCP(name="t", auth=provider)
    app = mcp.http_app(transport="http")
    _install(revoked={"gone"})
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://127.0.0.1:5039") as c:
            ok = await c.post("/mcp", headers={"Authorization": f"Bearer {tokens.make(jti='keep')}"},
                              json={})
            r = await c.post("/mcp", headers={"Authorization": f"Bearer {tokens.make(jti='gone')}"},
                             json={})
    assert ok.status_code != 401  # a live token gets past auth
    assert r.status_code == 401
    assert r.json()["error"] == "invalid_token"


# -- console ------------------------------------------------------------------------


def _console_app():
    from src.api.router.console_session import router
    from src.auth.console_session import ConsoleIdentity, require_console_session
    app = FastAPI()
    app.include_router(router)

    @app.get("/probe")
    async def probe(ident: ConsoleIdentity = Depends(require_console_session)):
        return ident.public_view()
    return app


def test_console_sign_in_refuses_a_revoked_pat(console_auth, tokens):
    _install(revoked={"gone"})
    c = TestClient(_console_app())
    r = c.post("/api/console/session", json={"token": tokens.make(jti="gone")}, headers=CSRF)
    assert r.status_code == 401 and r.json()["detail"]["error"] == REVOKED


def test_console_sign_in_fail_closed_is_503(console_auth, tokens):
    _install(fail_mode=FAIL_CLOSED, fresh=False)
    c = TestClient(_console_app())
    r = c.post("/api/console/session", json={"token": tokens.make()}, headers=CSRF)
    assert r.status_code == 503 and r.json()["detail"]["error"] == UNAVAILABLE


def test_revoking_the_pat_ends_the_console_session(console_auth, tokens):
    rl = _install()
    c = TestClient(_console_app())
    assert c.post("/api/console/session", json={"token": tokens.make(jti="pat-1")},
                  headers=CSRF).status_code == 200
    assert c.get("/probe").status_code == 200
    assert c.get("/api/console/session").status_code == 200
    rl._revoked["pat-1"] = 9_999_999_999
    for path in ("/probe", "/api/console/session"):
        r = c.get(path)
        assert r.status_code == 401 and r.json()["detail"]["error"] == REVOKED
    # bearer use of the same PAT is refused too
    r = c.get("/probe", headers={"Authorization": f"Bearer {tokens.make(jti='pat-1')}"})
    assert r.status_code == 401


def test_console_session_fail_closed_is_503(console_auth, tokens):
    rl = _install(fail_mode=FAIL_CLOSED)
    c = TestClient(_console_app())
    assert c.post("/api/console/session", json={"token": tokens.make()},
                  headers=CSRF).status_code == 200
    rl._last_success = time.time() - 10_000
    r = c.get("/probe")
    assert r.status_code == 503 and r.json()["detail"]["error"] == UNAVAILABLE


# -- observability ------------------------------------------------------------------


async def test_status_reports_last_poll_and_size():
    feed, clock = FakeFeed(), Clock()
    rl = _rl(feed, clock)
    rv.set_revocation_list(rl)
    feed.queue([("a", 9_999_999), ("b", 9_999_999)])
    await rl.poll_once()
    st = rv.revocation_status()
    assert st["status"] == "healthy" and st["revoked_count"] == 2 and st["enabled"] is True
    assert st["last_success"].startswith("1970-01-12T13:46:40")
    assert st["fail_mode"] == "open" and st["poll_interval_sec"] == 30.0 and st["stale"] is False
