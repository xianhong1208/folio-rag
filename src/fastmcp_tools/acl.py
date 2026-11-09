
"""ACL — derive the caller's owner key from the Authorization header and check folder access.

Design principles:
- Single entry point: the first line of every tool call is always
  `token = extract_token()`, followed by `verify_folder_access(token,
  folder_name)` to obtain the folder ORM object.
- ACL failures always raise InvalidTokenError / UnauthorizedAccessError (domain
  exceptions), which middleware converts uniformly into a standard MCP error
  response.
- ACL results are not cached here.

verify_folder_access lives in the domain layer (src/domain/rag/folder_acl.py,
with no fastmcp dependency). extract_token reads the MCP HTTP request context,
which is genuine delivery-layer logic, so it stays here.
"""

from __future__ import annotations

from typing import Optional

from fastmcp.server.dependencies import get_access_token, get_http_request

from src.auth.owner_key import TokenOwner, owner_from_bearer, owner_from_claims
from src.domain.exceptions import InvalidTokenError
from src.log import get_mcptools_logger

logger = get_mcptools_logger()


def _verified_claims(raw_token: str) -> Optional[dict]:
    """Claims of this request's token as verified by the FastMCP auth provider, or
    None (auth disabled, or the provider did not verify this exact token)."""
    try:
        access = get_access_token()
    except Exception:  # noqa: BLE001 - no auth context
        return None
    if access is None or getattr(access, "token", None) != raw_token:
        return None
    claims = getattr(access, "claims", None)
    return claims if isinstance(claims, dict) else None


def _extract_owner() -> tuple[TokenOwner, bool]:
    """(owner, verified) for the request's Bearer token."""
    request = get_http_request()
    auth_header = request.headers.get("Authorization", "")

    if not auth_header:
        raise InvalidTokenError("Missing Authorization header")

    if not auth_header.startswith("Bearer "):
        raise InvalidTokenError(
            "Invalid Authorization header format. Expected: Bearer <token>"
        )

    parts = auth_header.split(" ")
    if len(parts) != 2 or not parts[1]:
        raise InvalidTokenError("Token is empty or malformed")

    # Ownership key = the person the token belongs to (its sub), not the token.
    claims = _verified_claims(parts[1])
    if claims is not None:
        return owner_from_claims(claims), True
    return owner_from_bearer(parts[1]), False


def extract_token() -> str:
    """The caller's folder owner key (``"sub:<sub>"``) from the Bearer token.

    Raises:
        InvalidTokenError: header is missing or malformed, or the token has no ``sub``
    """
    return _extract_owner()[0].owner_key


async def extract_owner_key() -> str:
    """:func:`extract_token`, first claiming the token's legacy per-token folders for
    its person (src/adapter/folder_claim.py) — only when the FastMCP auth provider
    verified this token."""
    owner, verified = _extract_owner()
    if verified:
        from src.adapter.folder_claim import claim_for
        await claim_for(owner)
    return owner.owner_key
