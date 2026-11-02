"""Shared fixtures: locally signed MCP Gate-style tokens and console auth state.

No network: tokens are signed with an RSA key generated here, and the offline
verifier's JWKS client is replaced by a stub that serves that key.
"""

from __future__ import annotations

import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

ISSUER = "http://gate.test"
AUDIENCE = "http://127.0.0.1:5039/mcp"
KID = "test-key-1"


@pytest.fixture(scope="session")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="session")
def other_rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class TokenFactory:
    """Mint RS256 access tokens shaped like MCP Gate's."""

    def __init__(self, key, other_key):
        self.key = key
        self.other_key = other_key

    def make(self, *, scopes=("rag:read",), sub="alice", jti=None, exp_in=3600,
             aud=AUDIENCE, iss=ISSUER, kid=KID, key=None, drop=(), **extra) -> str:
        now = int(time.time())
        claims = {"iss": iss, "aud": aud, "sub": sub, "iat": now,
                  "exp": now + exp_in, "jti": jti or str(uuid.uuid4()),
                  "scope": " ".join(scopes)}
        claims.update(extra)
        for k in drop:
            claims.pop(k, None)
        headers = {"kid": kid} if kid else {}
        return jwt.encode(claims, key or self.key, algorithm="RS256", headers=headers)


@pytest.fixture()
def tokens(rsa_key, other_rsa_key):
    return TokenFactory(rsa_key, other_rsa_key)


class _StubJWKClient:
    def __init__(self, public_key):
        self._key = jwt.PyJWK.from_dict(
            {**jwt.algorithms.RSAAlgorithm.to_jwk(public_key, as_dict=True),
             "kid": KID, "alg": "RS256", "use": "sig"})

    def get_signing_key_from_jwt(self, token):
        kid = jwt.get_unverified_header(token).get("kid")
        if kid != KID:
            raise jwt.PyJWKClientError(f'Unable to find a signing key that matches: "{kid}"')
        return self._key


@pytest.fixture()
def gate_verifier(rsa_key):
    """The real McpGateTokenVerifier, with its JWKS client stubbed (no network)."""
    from src.auth.mcp_gate_auth import AuthSettings, McpGateTokenVerifier
    v = McpGateTokenVerifier(AuthSettings(issuer=ISSUER, base_url="http://127.0.0.1:5039",
                                          audience=AUDIENCE))
    v._jwk_client = _StubJWKClient(rsa_key.public_key())
    return v


@pytest.fixture()
def console_auth(gate_verifier, tmp_path):
    """Install an enabled ConsoleAuth (token mode) for the duration of a test."""
    from src.auth import console_session as cs
    auth = cs.build_console_auth(
        _cfg(enabled=True), verifier=gate_verifier,
        secret_path=tmp_path / cs.SECRET_FILE_NAME)
    cs.set_console_auth(auth)
    yield auth
    cs.set_console_auth(None)


@pytest.fixture()
def open_console_auth():
    """Install an open-mode (auth disabled) ConsoleAuth for the duration of a test."""
    from src.auth import console_session as cs
    auth = cs.build_console_auth(_cfg(enabled=False))
    cs.set_console_auth(auth)
    yield auth
    cs.set_console_auth(None)


def _cfg(*, enabled, **console):
    from types import SimpleNamespace
    from src.config.model import ConsoleAuthConfig
    return SimpleNamespace(auth=SimpleNamespace(enabled=enabled,
                                                console=ConsoleAuthConfig(**console)))


@pytest.fixture()
def make_cfg():
    return _cfg


def console_identity(owner="owner-tok", *, admin=False, sub="alice"):
    """A token-mode console identity owning folders bound to ``owner``."""
    from src.auth.console_session import ConsoleIdentity
    scopes = ("rag:admin",) if admin else ()
    return ConsoleIdentity(mode="token", via="cookie", jti=owner, sub=sub,
                           scopes=scopes, is_admin=admin, expires_at=None)


@pytest.fixture()
def as_identity():
    """Return a helper that overrides the console guards on an app with an identity."""
    from fastapi import HTTPException
    from src.auth.console_session import require_console_admin, require_console_session

    def _apply(app, ident):
        async def _session():
            return ident

        async def _admin():
            if not ident.is_admin:
                raise HTTPException(403, {"error": "admin_required"})
            return ident
        app.dependency_overrides[require_console_session] = _session
        app.dependency_overrides[require_console_admin] = _admin
        return app
    return _apply
