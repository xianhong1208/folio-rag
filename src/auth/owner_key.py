"""Derive the folder owner key from a verified bearer token.

Folders belong to a **person**, not to one token: the owner key is the token's
``sub`` claim, prefixed with ``sub:`` (``"sub:<sub>"``). Every token MCP Gate issues
to the same user — personal access tokens and OAuth access tokens alike — carries
that user's id as ``sub``, so they all see the same folders, and an OAuth client
keeps its folders across the 15-minute access-token rotation.

The ``sub:`` prefix keeps the new keys apart from the legacy per-token keys
(the token's ``jti``, used before 1.4.1) still stored on folders nobody has
claimed yet; see :mod:`src.adapter.folder_claim` for how those move over.

A token without a ``sub`` cannot own folders and is refused for folder operations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import jwt

from src.domain.exceptions import InvalidTokenError

OWNER_PREFIX = "sub:"
# Folders.user_token is VARCHAR(256); a longer sub is stored as a digest.
_OWNER_KEY_MAX = 256
_SUB_DIGEST_PREFIX = "sub:sha256:"

MISSING_SUB_MESSAGE = (
    "This token has no subject (sub claim). Folio keys folders on the person a token "
    "belongs to, so it cannot be used for folder operations. Use a token issued by "
    "MCP Gate for a user (a personal access token or an OAuth sign-in)."
)


class MissingSubjectError(InvalidTokenError):
    """The token carries no ``sub``, so it has no owner key (401)."""

    def __init__(self):
        super().__init__(MISSING_SUB_MESSAGE)


def owner_key_for_sub(sub: Optional[str]) -> str:
    """The owner key for a token subject: ``"sub:" + sub``.

    Raises MissingSubjectError for an empty / missing subject — never returns a
    shared fallback key, which would hand every sub-less token the same folders.
    """
    if sub is None or not str(sub).strip():
        raise MissingSubjectError()
    sub = str(sub)
    key = OWNER_PREFIX + sub
    if len(key) > _OWNER_KEY_MAX:
        return _SUB_DIGEST_PREFIX + hashlib.sha256(sub.encode()).hexdigest()
    return key


def is_person_key(owner_key: Optional[str]) -> bool:
    """True for a person-scoped key (``sub:…``); False for a legacy per-token key."""
    return bool(owner_key) and owner_key.startswith(OWNER_PREFIX)


@dataclass(frozen=True)
class TokenOwner:
    """Who a token belongs to: the owner key, plus the ``jti`` that may still own
    legacy folders (see :mod:`src.adapter.folder_claim`)."""

    owner_key: str
    sub: str
    jti: Optional[str] = None


def owner_from_claims(claims: dict) -> TokenOwner:
    """Owner of a token from its (already verified) claims; MissingSubjectError without ``sub``."""
    sub = claims.get("sub") if isinstance(claims, dict) else None
    key = owner_key_for_sub(sub)
    jti = claims.get("jti")
    return TokenOwner(owner_key=key, sub=str(sub), jti=str(jti) if jti else None)


def unverified_claims(token: str) -> dict:
    """Read a JWT's claims without checking the signature (the caller verified it).

    A non-JWT (opaque) token has no claims: an empty dict, which then has no ``sub``.
    """
    try:
        claims = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def owner_from_bearer(token: str) -> TokenOwner:
    """Owner of an already-verified bearer token (MissingSubjectError without ``sub``).

    The signature is verified upstream (the REST auth dependency, the FastMCP auth
    provider, the console sign-in) and intentionally not re-checked here.
    """
    return owner_from_claims(unverified_claims(token))


def owner_key_from_bearer(token: str) -> str:
    """The owner key (``"sub:<sub>"``) of an already-verified bearer token."""
    return owner_from_bearer(token).owner_key
