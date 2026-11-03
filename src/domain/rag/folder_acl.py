
"""Folder ACL — token-to-folder permission checks (domain layer).

verify_folder_access depends only on FolderDB + domain exceptions, not on fastmcp.
`extract_token` (which reads the MCP HTTP request context) is the actual delivery-layer
logic and stays in fastmcp_tools/acl.py.

Design principles:
- Enforce at the DB query layer: FolderDB.get always takes a user_token, so querying
  someone else's folder with your token returns empty → denied. This is not a
  "fetch first, then compare" approach.
- ACL failures always raise UnauthorizedAccessError (a domain exception); the caller's
  middleware converts it uniformly into a standard error response.
- Deliberately does not distinguish "folder does not exist" from "token lacks access",
  to avoid information leakage.
- ACL results are not cached here.
"""

from __future__ import annotations

from db.folderdb import FolderDB
from src.domain.exceptions import UnauthorizedAccessError
from src.log import get_mcptools_logger

logger = get_mcptools_logger()


def verify_folder_access(token: str, folder_name: str):
    """Verify the token may access this folder; return the Folder ORM object.

    Raises:
        UnauthorizedAccessError: the token cannot access this folder
    """
    folders = FolderDB.get(name=folder_name, user_token=token)
    if not folders:
        raise UnauthorizedAccessError(
            resource_type="folder",
            resource_id=folder_name,
            reason="Folder not found or token lacks access",
        )
    return folders[0]
