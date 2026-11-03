
"""ACL — extract the token from the Authorization header and check folder access.

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

from fastmcp.server.dependencies import get_http_request

from src.auth.owner_key import owner_key_from_bearer
from src.domain.exceptions import InvalidTokenError
from src.log import get_mcptools_logger

logger = get_mcptools_logger()


def extract_token() -> str:
    """Extract the Bearer token from the Authorization header.

    Raises:
        InvalidTokenError: header is missing or malformed
    """
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

    # Ownership key = this token's jti (per-token scope), not the raw token.
    return owner_key_from_bearer(parts[1])
