"""FastMCP ``JWTVerifier`` that also honours MCP Gate's revocation feed.

Governs the ``/mcp`` transport. Imported lazily by ``build_auth_provider`` so a
disabled-auth deployment never imports the FastMCP auth stack.
"""

from __future__ import annotations

from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.providers.jwt import JWTVerifier

from src.auth.revocation import check_revoked
from src.log import get_auth_logger

logger = get_auth_logger()


class RevocationAwareJWTVerifier(JWTVerifier):
    """Offline RS256 verification plus a lookup in the revocation list.

    Returning None makes FastMCP answer 401 ``invalid_token``, the same as for an
    expired or badly signed token.
    """

    async def load_access_token(self, token: str) -> AccessToken | None:
        result = await super().load_access_token(token)
        if result is None:
            return None
        jti = (result.claims or {}).get("jti")
        reason = check_revoked(str(jti) if jti else None)
        if reason:
            logger.warning(f"MCP bearer token rejected: {reason} (client {result.client_id}, "
                           f"jti {str(jti)[:8]}...)")
            return None
        return result
