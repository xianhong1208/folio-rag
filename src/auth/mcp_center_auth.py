"""Resource-server side of OAuth 2.1: verify MCP Center access tokens.

Agentic RAG never issues tokens. It trusts one authorization server (MCP Center),
fetches its JWKS once and verifies every bearer token OFFLINE. Two consumers use this
module:

- The REST API + MCP tool filter go through ``McpCenterTokenVerifier`` (PyJWT-based),
  registered via ``set_remote_verifier`` so the existing middleware / tool-filter keep
  working unchanged — they only rely on the ``TokenVerifyResult`` shape and the
  ``async verify(token)`` contract.
- The FastMCP ``/mcp`` transport uses ``build_auth_provider`` (FastMCP's
  ``RemoteAuthProvider`` + ``JWTVerifier``), which also publishes
  ``/.well-known/oauth-protected-resource/mcp`` so OAuth-capable MCP clients can
  discover where to sign in.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import jwt

from src.log import get_auth_logger
from src.auth.remote_auth import TokenVerifyResult

logger = get_auth_logger()


@dataclass(frozen=True)
class AuthSettings:
    """Resolved authentication settings (issuer, audience and public base URL)."""
    issuer: str
    base_url: str
    audience: str
    required_scopes: list[str] = field(default_factory=list)

    @property
    def jwks_uri(self) -> str:
        """JWKS endpoint published by the issuer (MCP Center)."""
        return f"{self.issuer}/.well-known/jwks.json"


def settings_from_config(config) -> Optional[AuthSettings]:
    """Return the effective auth settings for ``config.auth``, or None when disabled.

    Mirrors AnyDoc's ``resolve_auth_settings``: audience defaults to the server's own
    MCP URL (``http://<host>:<port>/mcp``) when left empty in config.
    """
    auth = getattr(config, "auth", None)
    if auth is None or not getattr(auth, "enabled", False):
        return None

    issuer = getattr(auth, "issuer", None)
    if not issuer:
        raise RuntimeError(
            "auth.issuer is required when auth.enabled is true "
            "(set it to the MCP Center base URL)"
        )

    server = getattr(config, "server", None)
    host = getattr(server, "host", "127.0.0.1")
    if host == "0.0.0.0":
        host = "127.0.0.1"  # 0.0.0.0 is a bind address, not a reachable audience URL
    port = getattr(server, "port", 8000)
    base_url = f"http://{host}:{port}".rstrip("/")

    audience = getattr(auth, "audience", None) or f"{base_url}/mcp"
    required_scopes = list(getattr(auth, "required_scopes", None) or [])

    return AuthSettings(
        issuer=str(issuer).rstrip("/"),
        base_url=base_url,
        audience=audience,
        required_scopes=required_scopes,
    )


def _parse_scopes(claims: dict) -> list[str]:
    """Normalise scopes from a JWT: RFC 8693 space-delimited ``scope`` or list ``scopes``."""
    raw = claims.get("scope")
    if isinstance(raw, str):
        return raw.split()
    scopes = claims.get("scopes")
    if isinstance(scopes, (list, tuple)):
        return list(scopes)
    if isinstance(scopes, str):
        return scopes.split()
    return []


class McpCenterTokenVerifier:
    """Offline RS256 JWT verifier for MCP Center access tokens.

    Exposes the SAME async interface as the legacy ``RemoteTokenVerifier``
    (``async verify(token) -> TokenVerifyResult`` plus a no-op ``close``), so the
    middleware, dependencies and MCP tool filter need no changes.

    Verification is done with PyJWT + ``jwt.PyJWKClient`` (the robust, well-documented
    path — no dependency on FastMCP internals). The PyJWKClient is created once and
    caches signing keys; the synchronous PyJWT calls run in a worker thread so
    ``verify`` stays non-blocking.
    """

    def __init__(self, settings: AuthSettings):
        self._settings = settings
        # PyJWKClient caches fetched keys internally; keep one instance alive.
        self._jwk_client = jwt.PyJWKClient(settings.jwks_uri)
        logger.info(
            f"McpCenterTokenVerifier initialized: jwks_uri={settings.jwks_uri} "
            f"audience={settings.audience}"
        )

    def _verify_sync(self, token: str) -> TokenVerifyResult:
        """Blocking RS256 verification (runs in a thread)."""
        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self._settings.audience,
                issuer=self._settings.issuer,
                options={"require": ["exp"]},
            )
        except jwt.PyJWTError as e:
            return TokenVerifyResult(valid=False, error=str(e))
        except Exception as e:  # noqa: BLE001 - JWKS fetch / network failures
            logger.warning(f"Token verification error: {e}")
            return TokenVerifyResult(valid=False, error=str(e))

        scopes = _parse_scopes(claims)

        # Enforce required scopes, if configured.
        if self._settings.required_scopes:
            missing = [s for s in self._settings.required_scopes if s not in scopes]
            if missing:
                return TokenVerifyResult(
                    valid=False,
                    error=f"insufficient_scope: missing {missing}",
                )

        return TokenVerifyResult(
            valid=True,
            user_name=claims.get("sub"),
            scopes=scopes,
            expires_at=claims.get("exp"),
        )

    async def verify(self, token: str) -> TokenVerifyResult:
        """Verify a bearer token offline. Non-blocking (PyJWT runs in a thread)."""
        return await asyncio.to_thread(self._verify_sync, token)

    def close(self) -> None:
        """No-op: kept for interface parity with RemoteTokenVerifier."""
        return None


def build_auth_provider(config):
    """Return the FastMCP auth provider for ``config.auth``, or None when disabled.

    Mirrors AnyDoc: a ``RemoteAuthProvider`` wrapping a ``JWTVerifier`` that validates
    RS256 tokens against the issuer's JWKS. This governs the FastMCP ``/mcp`` transport
    and publishes the RFC 9728 protected-resource metadata.
    """
    settings = settings_from_config(config)
    if settings is None:
        logger.warning("Authentication is DISABLED - every client can call the tools")
        return None

    # Imported lazily so a disabled-auth deployment need not import FastMCP auth stack.
    from fastmcp.server.auth import RemoteAuthProvider
    from fastmcp.server.auth.providers.jwt import JWTVerifier
    from pydantic import AnyHttpUrl

    provider = RemoteAuthProvider(
        token_verifier=JWTVerifier(
            jwks_uri=settings.jwks_uri,
            issuer=settings.issuer,
            audience=settings.audience,
            required_scopes=settings.required_scopes,
        ),
        authorization_servers=[AnyHttpUrl(settings.issuer)],
        base_url=settings.base_url,
    )
    logger.info(
        f"Authentication enabled: issuer={settings.issuer} audience={settings.audience}"
    )
    return provider
