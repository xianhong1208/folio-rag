"""Admin-console login: PAT -> stateless session cookie, plus the FastAPI guards.

The console signs in with an MCP Gate access token (a personal access token issued
for this service). The token is verified ONCE, with the same offline verifier the
REST API uses (signature, issuer, audience, expiry), and must carry a ``jti``: that
``jti`` is the per-token owner key every folder is bound to, so the console session
sees exactly the folders a REST / MCP caller holding that token sees.

The session itself is a short HS256-signed JWT in an HttpOnly cookie::

    {typ: "rag-console-session", jti, sub, scopes, iat, exp}

It never contains (or echoes) the raw token. ``exp`` is the earlier of the token's
own expiry and ``now + auth.console.session_hours``. The session is stateless, but
every request re-checks its ``jti`` against MCP Gate's revocation list (when
``auth.revocation`` is enabled), so revoking the token in MCP Gate also ends the
console sessions created from it within a poll interval.

Guards:

- ``require_console_session`` — any valid session cookie, or ``Authorization:
  Bearer <PAT>`` (for scripts; verified the same way). Cookie-authenticated unsafe
  requests must also pass the CSRF check (``X-RAG-Console: 1`` and a same-origin
  ``Origin`` / ``Referer``): other localhost ports are same-site, so SameSite=Strict
  alone does not stop them.
- ``require_console_admin`` — the above plus the admin scope (``rag:admin``).

With ``auth.enabled: false`` the console runs in *open* (dev) mode: no login, every
request is treated as an admin that can see every folder.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

import jwt
from fastapi import HTTPException, Request

from src.auth.revocation import UNAVAILABLE, check_revoked
from src.log import get_auth_logger

logger = get_auth_logger()

COOKIE_NAME = "rag_console_session"
SESSION_TYP = "rag-console-session"
CSRF_HEADER = "X-RAG-Console"
SECRET_FILE_NAME = ".console_session_secret"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# A pasted token larger than this is not an MCP Gate access token.
MAX_TOKEN_CHARS = 16384

# Human-readable explanations for the verifier's error codes (shown on the login screen).
_REJECTIONS: dict[str, str] = {
    "token_expired": "This token has expired. Issue a new one in MCP Gate.",
    "wrong_audience": (
        "This token was issued for a different service (audience mismatch). "
        "Issue a token for Folio in MCP Gate."
    ),
    "wrong_issuer": "This token was not issued by the MCP Gate this server trusts.",
    "bad_signature": "The token's signature is invalid.",
    "unknown_signing_key": (
        "The token was signed with a key this server does not know "
        "(not issued by the trusted MCP Gate, or its keys were rotated)."
    ),
    "malformed_token": "That is not a valid access token.",
    "missing_claim": "The token is missing a required claim.",
    "missing_jti": (
        "The token has no jti (token ID), which the console needs as the folder "
        "owner key. Use an MCP Gate personal access token."
    ),
    "insufficient_scope": "The token lacks a scope this server requires.",
    "jwks_unavailable": "Cannot reach MCP Gate to fetch its signing keys. Try again shortly.",
    "token_revoked": "This token has been revoked in MCP Gate. Issue a new one and sign in again.",
    "revocation_unavailable": (
        "Cannot confirm with MCP Gate that this token is still valid (its revocation feed is "
        "unreachable). Try again shortly."
    ),
    "invalid_token": "The token could not be verified.",
}


def _auth_error(status: int, code: str, description: Optional[str] = None) -> HTTPException:
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    return HTTPException(
        status_code=status,
        detail={"error": code, "error_description": description or _REJECTIONS.get(code, code)},
        headers=headers,
    )


@dataclass(frozen=True)
class ConsoleIdentity:
    """Who is using the console, resolved from a session cookie or a bearer PAT."""

    mode: str  # "token" (auth enabled) | "open" (auth disabled, dev mode)
    via: str  # "cookie" | "bearer" | "open"
    jti: Optional[str] = None
    sub: Optional[str] = None
    scopes: tuple[str, ...] = ()
    is_admin: bool = False
    expires_at: Optional[int] = None

    @property
    def is_open(self) -> bool:
        return self.mode == "open"

    @property
    def owner_key(self) -> Optional[str]:
        """Folder owner key to scope queries by; None in open mode (every folder)."""
        return None if self.is_open else self.jti

    def owns(self, folder_owner: Optional[str]) -> bool:
        """True when this identity may operate on a folder bound to ``folder_owner``.

        Fail-closed: an ownerless folder is nobody's (open mode excepted).
        """
        if self.is_open:
            return True
        return bool(folder_owner) and folder_owner == self.jti

    @property
    def jti_prefix(self) -> Optional[str]:
        return f"{self.jti[:8]}…" if self.jti else None

    @property
    def actor(self) -> str:
        """Audit name for changes made through the console (never the raw token)."""
        if self.is_open:
            return "console (open mode)"
        return self.sub or (f"jti:{self.jti_prefix}" if self.jti else "console")

    def public_view(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "via": self.via,
            "jti_prefix": self.jti_prefix,
            "sub": self.sub,
            "scopes": list(self.scopes),
            "is_admin": self.is_admin,
            "expires_at": self.expires_at,
        }


@dataclass
class ConsoleAuth:
    """Console authentication state for one server instance."""

    enabled: bool
    verifier: Any = None  # McpGateTokenVerifier-compatible: async verify(token)
    secret: bytes = b""
    session_hours: float = 8.0
    secure_cookie: Any = "auto"  # True | False | "auto"
    admin_scope: str = "rag:admin"
    clock: Any = field(default=time.time, repr=False)

    # -- identity ---------------------------------------------------------------

    def open_identity(self) -> ConsoleIdentity:
        return ConsoleIdentity(mode="open", via="open", is_admin=True)

    def _identity(self, *, via: str, jti: str, sub: Optional[str], scopes, exp) -> ConsoleIdentity:
        scopes = tuple(str(s) for s in (scopes or ()))
        return ConsoleIdentity(
            mode="token", via=via, jti=str(jti), sub=sub or None, scopes=scopes,
            is_admin=self.admin_scope in scopes,
            expires_at=int(exp) if exp is not None else None,
        )

    async def verify_pat(self, token: str) -> ConsoleIdentity:
        """Verify an MCP Gate access token; raise 401/503 with a reason on failure."""
        token = (token or "").strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not token:
            raise _auth_error(401, "login_required", "Paste an access token to sign in.")
        if len(token) > MAX_TOKEN_CHARS:
            raise _auth_error(401, "malformed_token")
        if self.verifier is None:
            raise _auth_error(503, "auth_not_configured", "Token verification is not configured.")
        result = await self.verifier.verify(token)
        if not result.valid:
            code = getattr(result, "error_code", None) or "invalid_token"
            logger.warning(f"Console token rejected: {code}")
            raise _auth_error(503 if code in ("jwks_unavailable", UNAVAILABLE) else 401, code)
        if not getattr(result, "jti", None):
            raise _auth_error(401, "missing_jti")
        return self._identity(via="bearer", jti=result.jti, sub=result.user_name,
                              scopes=result.scopes, exp=result.expires_at)

    # -- session cookie ---------------------------------------------------------

    def issue(self, ident: ConsoleIdentity) -> tuple[str, int]:
        """Sign a session for ``ident``; return (cookie value, exp epoch seconds)."""
        now = int(self.clock())
        exp = now + int(self.session_hours * 3600)
        if ident.expires_at is not None:
            exp = min(exp, int(ident.expires_at))
        payload = {
            "typ": SESSION_TYP,
            "jti": ident.jti,
            "sub": ident.sub,
            "scopes": list(ident.scopes),
            "iat": now,
            "exp": exp,
        }
        return jwt.encode(payload, self.secret, algorithm="HS256"), exp

    def read_session(self, value: str) -> ConsoleIdentity:
        """Decode a session cookie; raise 401 when it is invalid or expired."""
        try:
            claims = jwt.decode(
                value, self.secret, algorithms=["HS256"],
                options={"require": ["exp", "iat", "jti"], "verify_exp": False, "verify_iat": False},
            )
            # Expiry is checked against our own clock (testable; PyJWT uses wall time).
            if int(claims["exp"]) <= int(self.clock()):
                raise jwt.ExpiredSignatureError("expired")
        except jwt.ExpiredSignatureError:
            raise _auth_error(401, "session_expired", "Your console session has expired. Sign in again.")
        except jwt.PyJWTError:
            raise _auth_error(401, "invalid_session", "Your console session is not valid. Sign in again.")
        if claims.get("typ") != SESSION_TYP or not claims.get("jti"):
            raise _auth_error(401, "invalid_session", "Your console session is not valid. Sign in again.")
        return self._identity(via="cookie", jti=claims["jti"], sub=claims.get("sub"),
                              scopes=claims.get("scopes") or (), exp=claims["exp"])

    def cookie_secure(self, request: Request) -> bool:
        if isinstance(self.secure_cookie, bool):
            return self.secure_cookie
        return request.url.scheme == "https"

    def set_cookie(self, response, request: Request, value: str, exp: int) -> None:
        response.set_cookie(
            COOKIE_NAME, value,
            max_age=max(int(exp - self.clock()), 0),
            path="/", httponly=True, samesite="strict",
            secure=self.cookie_secure(request),
        )

    def clear_cookie(self, response, request: Request) -> None:
        response.delete_cookie(
            COOKIE_NAME, path="/", httponly=True, samesite="strict",
            secure=self.cookie_secure(request),
        )


# -- CSRF -----------------------------------------------------------------------


def _origin_of(url: str) -> Optional[str]:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}".lower()


def check_csrf(request: Request) -> None:
    """Reject a cookie-authenticated unsafe request that did not come from this origin.

    Requires the ``X-RAG-Console: 1`` header (a cross-origin page cannot add it
    without a CORS preflight this server never grants) AND an ``Origin`` — or, when
    the browser omits it, a ``Referer`` — equal to the request's own origin.
    """
    if request.headers.get(CSRF_HEADER) != "1":
        raise _auth_error(403, "csrf_failed", f"Missing {CSRF_HEADER} header.")
    own = f"{request.url.scheme}://{request.url.netloc}".lower()
    origin = request.headers.get("origin")
    source = _origin_of(origin) if origin else None
    if source is None and not origin:
        referer = request.headers.get("referer")
        source = _origin_of(referer) if referer else None
    if source != own:
        raise _auth_error(403, "csrf_failed", "Cross-origin request refused.")


# -- module state + FastAPI guards ------------------------------------------------

_console_auth: Optional[ConsoleAuth] = None


def set_console_auth(auth: Optional[ConsoleAuth]) -> None:
    global _console_auth
    _console_auth = auth


def get_console_auth() -> ConsoleAuth:
    if _console_auth is None:
        raise _auth_error(503, "auth_not_configured", "Console authentication is not initialized.")
    return _console_auth


def _bearer(request: Request) -> Optional[str]:
    header = request.headers.get("authorization") or ""
    if header[:7].lower() == "bearer ":
        return header[7:].strip()
    return None


async def resolve_identity(request: Request, *, enforce_csrf: bool = True) -> ConsoleIdentity:
    """Resolve the caller from a Bearer PAT or the session cookie (in that order)."""
    auth = get_console_auth()
    if not auth.enabled:
        return auth.open_identity()
    token = _bearer(request)
    if token is not None:
        return await auth.verify_pat(token)
    value = request.cookies.get(COOKIE_NAME)
    if not value:
        raise _auth_error(401, "login_required", "Sign in to use the console.")
    ident = auth.read_session(value)
    # Sessions are stateless: re-check the token they came from on every request, so
    # revoking it in MCP Gate signs the console out too.
    reason = check_revoked(ident.jti)
    if reason:
        logger.warning(f"Console session rejected: {reason} (jti {ident.jti_prefix})")
        raise _auth_error(503 if reason == UNAVAILABLE else 401, reason)
    if enforce_csrf and request.method.upper() not in _SAFE_METHODS:
        check_csrf(request)
    return ident


async def require_console_session(request: Request) -> ConsoleIdentity:
    """FastAPI dependency: any signed-in console user (or a bearer PAT)."""
    return await resolve_identity(request)


async def require_console_admin(request: Request) -> ConsoleIdentity:
    """FastAPI dependency: a console user whose token carries the admin scope."""
    ident = await resolve_identity(request)
    if not ident.is_admin:
        scope = get_console_auth().admin_scope
        raise _auth_error(403, "admin_required",
                          f"This action needs a token with the '{scope}' scope.")
    return ident


# -- construction -----------------------------------------------------------------


def _derive_key(secret: str | bytes) -> bytes:
    raw = secret.encode() if isinstance(secret, str) else secret
    # Domain-separated, fixed-length HMAC key regardless of the configured secret's length.
    return hmac.new(b"agentic-rag-console-session", raw, hashlib.sha256).digest()


def default_secret_path() -> Path:
    from src.storage.file_storage import FileStorage
    return Path(FileStorage.STORAGE_ROOT) / SECRET_FILE_NAME


def load_or_create_secret(path: Path) -> bytes:
    """Read the persisted session secret, generating it (0600) on first use."""
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing.encode()
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_urlsafe(48)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value + "\n")
    logger.info(f"Generated a console session secret at {path}")
    return value.encode()


def build_console_auth(config, verifier=None, secret_path: Optional[Path] = None) -> ConsoleAuth:
    """Build the console auth state from ``config.auth`` (open mode when auth is disabled)."""
    auth_cfg = getattr(config, "auth", None)
    enabled = bool(getattr(auth_cfg, "enabled", False))
    console = getattr(auth_cfg, "console", None)
    session_hours = float(getattr(console, "session_hours", 8) or 8)
    secure_cookie = getattr(console, "secure_cookie", "auto")
    admin_scope = getattr(console, "admin_scope", None) or "rag:admin"
    if not enabled:
        logger.warning(
            "Console authentication is DISABLED (auth.enabled: false): the admin console "
            "at /admin is open to anyone who can reach this port, with full admin rights.")
        return ConsoleAuth(enabled=False, session_hours=session_hours,
                           secure_cookie=secure_cookie, admin_scope=admin_scope)
    configured = (getattr(console, "session_secret", None) or "").strip()
    raw = configured.encode() if configured else load_or_create_secret(
        secret_path or default_secret_path())
    return ConsoleAuth(enabled=True, verifier=verifier, secret=_derive_key(raw),
                       session_hours=session_hours, secure_cookie=secure_cookie,
                       admin_scope=admin_scope)
