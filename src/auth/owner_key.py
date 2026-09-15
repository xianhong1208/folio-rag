"""Derive a per-token owner key from a verified bearer token.

Folder/file ownership is keyed on this value. The design is **per-token**: two
different tokens own two different sets of folders, even when issued to the same
user. The key is therefore the JWT ``jti`` (JWT ID) — unique per token and short
enough for the ``user_token`` column — NOT the ``sub`` (which is shared across all
of a user's tokens) and NOT the raw access token (which is ~900 chars and would
overflow the column).

Implication of per-token ownership: if a token is re-issued (an OAuth access token
rotating, or a PAT being regenerated), its ``jti`` changes and the new token sees a
fresh, empty scope — the folders bound to the old ``jti`` are no longer reachable
by it. This is intended; pair long-lived folders with a long-lived token.
"""

import hashlib

import jwt


def owner_key_from_bearer(token: str) -> str:
    """Return the per-token owner key for an already-verified bearer token.

    Uses the JWT ``jti`` claim. The signature is verified upstream (the REST auth
    middleware and the FastMCP auth provider both run before this), so it is
    intentionally not re-checked here; we only read the claim. For a token with no
    ``jti`` — or a non-JWT/opaque token — fall back to a short stable hash so the
    value stays per-token and never overflows the owner column.
    """
    try:
        claims = jwt.decode(
            token, options={"verify_signature": False, "verify_aud": False}
        )
        jti = claims.get("jti")
        if jti:
            return str(jti)
    except Exception:
        pass
    return hashlib.sha256(token.encode()).hexdigest()
