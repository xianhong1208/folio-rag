
"""Reusable FastAPI Depends() dependencies for authentication and authorization."""

from uuid import UUID
from fastapi import Request, Depends
from typing import Optional

from db.cached_folderdb import CachedFolderDB
from db.filedb import FileDB
from db.db import Folder, File
from src.auth.owner_key import owner_key_from_bearer
from src.domain.exceptions import (
    InvalidTokenError,
    FolderNotFoundError,
    UnauthorizedAccessError,
    RAGFileNotFoundError
)


async def extract_token(request: Request) -> str:
    """Extract and validate the Bearer token from the Authorization header.

    Raises InvalidTokenError if the token is missing or malformed.
    """
    auth_header = request.headers.get('Authorization', '')

    if not auth_header:
        raise InvalidTokenError("Missing Authorization header")

    if not auth_header.startswith('Bearer '):
        raise InvalidTokenError("Invalid Authorization header format. Expected: Bearer <token>")

    parts = auth_header.split(' ')
    if len(parts) != 2:
        raise InvalidTokenError("Invalid Authorization header format")

    token = parts[1]

    if not token:
        raise InvalidTokenError("Token is empty")

    # Ownership key = this token's jti (per-token scope), not the raw token.
    return owner_key_from_bearer(token)


async def get_folder_by_id(
    folder_id: int,
    token: str = Depends(extract_token)
) -> Folder:
    """Get a folder by ID and verify the token owns it.

    Raises FolderNotFoundError if missing, UnauthorizedAccessError if not owned.
    """
    folder = CachedFolderDB.get_by_id(folder_id)

    if not folder:
        raise FolderNotFoundError(folder_id=folder_id)

    # Fail-closed: an ownerless folder (user_token NULL/empty) is always denied —
    # "no owner" means "nobody", not "anyone".
    if folder.user_token != token:
        raise UnauthorizedAccessError(
            resource_type="folder",
            resource_id=folder_id,
            reason="This folder belongs to another user"
        )

    return folder


async def get_folder_by_name(
    folder_name: str,
    token: str = Depends(extract_token)
) -> Folder:
    """Get a folder by name, scoped to the token's own folders.

    Raises FolderNotFoundError if no such folder is owned by the token.
    """
    # Lookup is already scoped by token, so a miss covers both "does not exist"
    # and "owned by someone else" — neither is distinguishable to the caller.
    folder = CachedFolderDB.get_by_name_and_token(folder_name, token)

    if not folder:
        raise FolderNotFoundError(folder_name=folder_name)

    return folder


async def get_file_by_id(
    file_id: UUID,
    token: str = Depends(extract_token)
) -> File:
    """Get a file by ID and verify the token owns its folder.

    Raises RAGFileNotFoundError if the file is missing, FolderNotFoundError if its
    folder is missing, UnauthorizedAccessError if the folder is not owned.
    """
    files = FileDB.get(id=file_id)

    if not files or len(files) == 0:
        raise RAGFileNotFoundError(file_id=file_id)

    file = files[0]

    folder = CachedFolderDB.get_by_id(file.folder_id)

    if not folder:
        raise FolderNotFoundError(folder_id=file.folder_id)

    # Fail-closed: as in get_folder_by_id, an ownerless folder is always denied.
    if folder.user_token != token:
        raise UnauthorizedAccessError(
            resource_type="file",
            resource_id=file_id,
            reason="This file belongs to a folder owned by another user"
        )
    
    return file
