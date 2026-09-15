"""Unit tests for src/auth (model.py + remote_auth.py).

HTTP is mocked (no network); cache TTL uses a monkeypatched fake clock.
"""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

import src.auth.remote_auth as remote_auth_module
from src.auth.model import (
    TokenCreateRequest,
    TokenResponse,
    TokenInfo,
    TokenListResponse,
)
from src.auth.remote_auth import (
    RemoteTokenVerifier,
    TokenVerifyResult,
    get_remote_verifier,
    reset_remote_verifier,
)


def _make_verifier(**kwargs) -> RemoteTokenVerifier:
    """Build a verifier that never hits the real network (the URL is fake, the HTTP client is mocked later)"""
    defaults = dict(timeout=0.1, retry_count=2, cache_ttl=60)
    defaults.update(kwargs)
    return RemoteTokenVerifier("http://token-server.invalid:1/", **defaults)


def _mock_response(status_code: int, json_data: dict | None = None) -> MagicMock:
    """Build a fake httpx Response (carrying only status_code and .json())"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def _install_mock_client(verifier: RemoteTokenVerifier) -> MagicMock:
    """Replace the verifier's persistent sync client with a MagicMock, return the mock client"""
    client = MagicMock()
    verifier._sync_client = client
    return client


# auth/model.py -- Pydantic models

def test_token_create_request_defaults():
    """TokenCreateRequest built with no arguments uses all defaults (TC-auth-01)"""
    req = TokenCreateRequest()
    assert req.user_name == "generated_user"
    assert req.scopes == ["read", "write", "admin"]
    assert req.length == 24
    assert req.expires_in_days is None


def test_token_create_request_explicit_none_rejected():
    """Explicitly passing None for expires_in_days -> ValidationError (the field type is int, not Optional[int]) (TC-auth-02)

    Note: the default None bypasses validation so TokenCreateRequest() passes,
    but explicitly passing None fails. This is the current behavior; the field
    declaration probably should be Optional[int].
    """
    with pytest.raises(ValidationError):
        TokenCreateRequest(expires_in_days=None)


def test_token_response_required_fields():
    """TokenResponse all six fields required; a missing field -> ValidationError (TC-auth-03)"""
    resp = TokenResponse(
        token="tok-abc", user_name="u1", scopes=["read"],
        expires_at=1780000000, expires_at_readable="2026-06-01 00:00:00",
        message="created",
    )
    assert resp.token == "tok-abc"
    with pytest.raises(ValidationError) as exc_info:
        TokenResponse(token="tok-abc")
    missing = {e["loc"][0] for e in exc_info.value.errors()}
    assert missing == {"user_name", "scopes", "expires_at", "expires_at_readable", "message"}


def test_token_info_roundtrip():
    """After building TokenInfo, model_dump preserves all field values (TC-auth-04)"""
    info = TokenInfo(
        user_name="u1", scopes=["read"], expires_at=1780000000,
        expires_at_readable="2026-06-01 00:00:00", module=["rag"],
    )
    dumped = info.model_dump()
    assert dumped["user_name"] == "u1"
    assert dumped["module"] == ["rag"]


def test_token_list_response_nested_coercion():
    """TokenListResponse.tokens nested dicts are auto-coerced into TokenInfo (TC-auth-05)"""
    resp = TokenListResponse(
        tokens={
            "tok-1": {
                "user_name": "u1", "scopes": ["read"], "expires_at": 1,
                "expires_at_readable": "x", "module": ["rag"],
            }
        },
        source="db", total_count=1, modules=["rag"],
    )
    assert isinstance(resp.tokens["tok-1"], TokenInfo)
    assert resp.tokens["tok-1"].user_name == "u1"


# remote_auth.py -- cache key

def test_get_cache_key_is_sha256():
    """_get_cache_key returns the token's SHA-256 hexdigest (64 hex, deterministic) (TC-auth-06)"""
    verifier = _make_verifier()
    key = verifier._get_cache_key("my-secret-token")
    assert key == hashlib.sha256(b"my-secret-token").hexdigest()
    assert len(key) == 64
    assert key == verifier._get_cache_key("my-secret-token")  # same token, same key


def test_get_cache_key_no_plaintext_and_distinct():
    """Different token -> different key; the key does not contain the plaintext token (TC-auth-07)"""
    verifier = _make_verifier()
    key_a = verifier._get_cache_key("token-aaaa")
    key_b = verifier._get_cache_key("token-bbbb")
    assert key_a != key_b
    assert "token-aaaa" not in key_a


# remote_auth.py -- cache TTL (monkeypatched fake clock)

@pytest.fixture
def fake_clock(monkeypatch):
    """Replace the time module inside remote_auth with a controllable fake clock"""
    clock = SimpleNamespace(now=1_000_000.0)
    monkeypatch.setattr(
        remote_auth_module, "time", SimpleNamespace(time=lambda: clock.now)
    )
    return clock


def test_cache_hit_within_ttl(fake_clock):
    """Within TTL, _get_from_cache hits and returns the original TokenVerifyResult (TC-auth-08)"""
    verifier = _make_verifier(cache_ttl=60)
    result = TokenVerifyResult(valid=True, user_name="u1")
    verifier._set_cache("key1", result)
    fake_clock.now += 59  # not yet expired
    assert verifier._get_from_cache("key1") is result


def test_cache_expired_after_ttl(fake_clock):
    """Past TTL -> returns None and the expired entry is removed (TC-auth-09)"""
    verifier = _make_verifier(cache_ttl=60)
    verifier._set_cache("key1", TokenVerifyResult(valid=True))
    fake_clock.now += 61  # expired
    assert verifier._get_from_cache("key1") is None
    assert "key1" not in verifier._cache  # already evicted


def test_clear_cache(fake_clock):
    """clear_cache empties all cache entries (TC-auth-10)"""
    verifier = _make_verifier()
    verifier._set_cache("k1", TokenVerifyResult(valid=True))
    verifier._set_cache("k2", TokenVerifyResult(valid=True))
    verifier.clear_cache()
    assert verifier._cache == {}


# remote_auth.py -- verify() across HTTP paths (mock httpx, no network)

async def test_verify_success_and_cached(fake_clock):
    """200 + valid:true -> success result cached, so a second verify does not call the API (TC-auth-11)"""
    verifier = _make_verifier()
    client = _install_mock_client(verifier)
    client.post.return_value = _mock_response(
        200,
        {"valid": True, "user_name": "u1", "scopes": ["read"], "expires_at": 123},
    )

    result = await verifier.verify("good-token")
    assert result.valid is True
    assert result.user_name == "u1"
    assert result.scopes == ["read"]
    assert result.expires_at == 123
    assert client.post.call_count == 1

    # Second call hits the cache, post is not called again
    result2 = await verifier.verify("good-token")
    assert result2.valid is True
    assert client.post.call_count == 1


async def test_verify_invalid_result_not_cached():
    """200 + valid:false -> not cached, so a second verify still calls the API (TC-auth-12)"""
    verifier = _make_verifier()
    client = _install_mock_client(verifier)
    client.post.return_value = _mock_response(200, {"valid": False, "error": "expired"})

    result = await verifier.verify("bad-token")
    assert result.valid is False
    assert result.error == "expired"

    await verifier.verify("bad-token")
    assert client.post.call_count == 2  # no cache -> calls every time


async def test_verify_401_no_retry():
    """4xx client error -> fail immediately without retry, error carries the status code (TC-auth-13)"""
    verifier = _make_verifier(retry_count=2)
    client = _install_mock_client(verifier)
    client.post.return_value = _mock_response(401)

    result = await verifier.verify("unauthorized-token")
    assert result.valid is False
    assert result.error == "token_server_error_401"
    assert client.post.call_count == 1  # 4xx is not retried


async def test_verify_timeout_retries_then_fails():
    """timeout -> retried retry_count+1 times then fails, error='timeout' (TC-auth-14)"""
    verifier = _make_verifier(retry_count=2)
    client = _install_mock_client(verifier)
    client.post.side_effect = httpx.TimeoutException("boom")

    result = await verifier.verify("any-token")
    assert result.valid is False
    assert result.error == "timeout"
    assert client.post.call_count == 3  # 1 original + 2 retries


async def test_verify_5xx_retries_then_fails():
    """5xx server error -> fails after retries are exhausted, error carries the status code (TC-auth-15)"""
    verifier = _make_verifier(retry_count=1)
    client = _install_mock_client(verifier)
    client.post.return_value = _mock_response(503)

    result = await verifier.verify("any-token")
    assert result.valid is False
    assert result.error == "token_server_error_503"
    assert client.post.call_count == 2


async def test_verify_connect_error():
    """Connection failure (ConnectError) -> error='connection_error' (TC-auth-16)"""
    verifier = _make_verifier(retry_count=0)
    client = _install_mock_client(verifier)
    client.post.side_effect = httpx.ConnectError("refused")

    result = await verifier.verify("any-token")
    assert result.valid is False
    assert result.error == "connection_error"


def test_verify_sync_success_with_service_identity():
    """verify_sync synchronous path succeeds, payload carries token and service host/port (TC-auth-17)"""
    verifier = _make_verifier(service_host="10.0.0.1", service_port=8080)
    client = _install_mock_client(verifier)
    client.post.return_value = _mock_response(200, {"valid": True, "user_name": "u1"})

    result = verifier.verify_sync("sync-token")
    assert result.valid is True
    payload = client.post.call_args.kwargs["json"]
    assert payload == {
        "token": "sync-token",
        "service_host": "10.0.0.1",
        "service_port": 8080,
    }


def test_get_remote_verifier_singleton_lifecycle():
    """get_remote_verifier on first call without a URL -> ValueError; with a URL returns a singleton (TC-auth-18)"""
    reset_remote_verifier()
    try:
        with pytest.raises(ValueError, match="token_server_url is required"):
            get_remote_verifier()
        v1 = get_remote_verifier("http://token-server.invalid:1")
        v2 = get_remote_verifier()  # no need to pass the URL again afterward
        assert v1 is v2
        assert v1.token_server_url == "http://token-server.invalid:1"  # trailing slash stripped
    finally:
        reset_remote_verifier()


def test_sync_client_lazy_init_and_reuse():
    """_get_sync_client: created on first use, reused thereafter (lazy init)."""
    from src.auth.remote_auth import RemoteTokenVerifier
    v = RemoteTokenVerifier(token_server_url="http://localhost:1")
    c1 = v._get_sync_client()
    assert v._get_sync_client() is c1
    v.close()
